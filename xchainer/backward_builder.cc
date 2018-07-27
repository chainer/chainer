#include "xchainer/backward_builder.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

using internal::ArrayNode;
using internal::OpNode;

}  // namespace

RetainedOutputToken::RetainedOutputToken(internal::ArrayBody::Params output_array_params, size_t output_index)
    : output_array_params_{std::move(output_array_params)}, output_index_{output_index} {}

BackwardBuilder::BackwardBuilder(const char* op_name, std::initializer_list<ConstArrayRef> outputs)
    : op_name_{op_name}, outputs_{outputs.begin(), outputs.end()} {
    // Outputs requiring grad (e.g. in-place ops.) must have been detected and reported before reaching here.
    assert(std::all_of(
            outputs.begin(), outputs.end(), [](const Array& output) { return internal::GetArrayBody(output)->nodes().empty(); }));
    // All output arrays must have the same device.
    assert(std::all_of(outputs.begin(), outputs.end(), [&outputs](const Array& output) {
        return &outputs.begin()->get().device() == &output.device();
    }));
}

BackwardBuilder::Target::Target(BackwardBuilder& builder, std::initializer_list<ConstArrayRef> inputs)
    : builder_{builder}, inputs_{inputs.begin(), inputs.end()} {
    // All input arrays must have the same device.
    assert(std::all_of(
            inputs.begin(), inputs.end(), [&inputs](const Array& input) { return &input.device() == &(inputs.begin()->get().device()); }));

    PrepareGraphToNextArrayNodes();
}

// Collect input ArrayNodes, grouped by graph considering IsBackpropRequired
void BackwardBuilder::Target::PrepareGraphToNextArrayNodes() {
    assert(graph_to_next_array_nodes_.empty());
    // TODO(niboshi): Probably linear search with a simple vector is faster than hash table.
    for (size_t i_input = 0; i_input < inputs_.size(); ++i_input) {
        const Array& input = *(inputs_.begin() + i_input);
        for (std::shared_ptr<ArrayNode>& next_array_node : internal::GetArrayBody(input)->nodes()) {
            const GraphId& graph_id = next_array_node->graph_id();
            if (!IsBackpropRequired(graph_id, input.device().context())) {
                continue;
            }

            // Add the array node to the mapping
            auto insert_result = graph_to_next_array_nodes_.emplace(graph_id, NextArrayNodes{});
            auto& vec = insert_result.first->second;
            if (insert_result.second) {
                // New array node for a graph. Fill all array nodes with nullptr.
                vec.resize(inputs_.size());
            }
            // Assign valid pointer to the array node.
            vec[i_input] = &next_array_node;
        }
    }

#ifndef NDEBUG
    for (auto& pair : graph_to_next_array_nodes_) {
        const GraphId& graph_id = pair.first;
        const NextArrayNodes& vec = pair.second;
        for (const std::shared_ptr<ArrayNode>* array_node : vec) {
            assert(array_node == nullptr || graph_id == (*array_node)->graph_id());
        }
    }
#endif  // NDEBUG
}

// Create an op node for a specific graph.
// Edges from output nodes to the op node are connected.
std::shared_ptr<OpNode>& BackwardBuilder::Target::FindOrCreateOpNode(const GraphId& graph_id) {
    // Find op node
    auto insert_result = op_node_map().emplace(graph_id, nullptr);
    if (insert_result.second) {
        insert_result.first->second = OpNode::CreateWithPrevArrayNodes(op_name(), graph_id, outputs());
    }
    assert(!op_node_map().empty());
    return insert_result.first->second;
}

// Add shared ptrs from the op nodes to previous array nodes of outer graphs.
void BackwardBuilder::Target::RegisterOuterGraphsPreviousArrayNodes(const std::vector<OpNode*>& op_nodes) {
    if (op_nodes.size() < 2) {  // op_nodes.size() is the number of graphs
        return;
    }

    std::unordered_map<GraphId, std::vector<std::shared_ptr<ArrayNode>>> prev_array_node_all_graphs;
    for (const Array& output : outputs()) {
        for (const std::shared_ptr<ArrayNode>& output_array_node : internal::GetArrayBody(output)->nodes()) {
            prev_array_node_all_graphs[output_array_node->graph_id()].emplace_back(output_array_node);
        }
    }

    for (OpNode* op_node : op_nodes) {
        for (const auto& tup : prev_array_node_all_graphs) {
            assert(tup.second.size() == outputs().size());
            if (tup.first >= op_node->graph_id()) {
                continue;
            }
            op_node->RegisterOuterGraphsPreviousArrayNodes(tup.first, tup.second);
        }
    }
}

void BackwardBuilder::Target::Define(const BackwardFunction& backward_func) {
    // Pointers to op nodes involved in this backward function
    std::vector<OpNode*> op_nodes;

    // Create op node for each graph
    for (const auto& pair : graph_to_next_array_nodes_) {
        const GraphId& graph_id = pair.first;
        const NextArrayNodes& next_array_nodes = pair.second;

        std::shared_ptr<OpNode>& op_node = FindOrCreateOpNode(graph_id);

        // Keep the list of op nodes involved in this backward function
        op_nodes.emplace_back(op_node.get());

        // Add edges to the input nodes
        std::vector<std::shared_ptr<ArrayNode>> temp_next_array_nodes;
        temp_next_array_nodes.reserve(next_array_nodes.size());
        std::transform(
                next_array_nodes.begin(),
                next_array_nodes.end(),
                std::back_inserter(temp_next_array_nodes),
                [](const std::shared_ptr<ArrayNode>* array_node) { return array_node == nullptr ? nullptr : *array_node; });
        op_node->RegisterBackwardFunction(std::move(temp_next_array_nodes), backward_func);
    }

    if (!any_defined()) {
        // TODO(niboshi): Do this only when BackwardBuilder::RetainOutput() is called.
        RegisterOuterGraphsPreviousArrayNodes(op_nodes);
    }
    set_any_defined(true);
}

RetainedOutputToken BackwardBuilder::RetainOutput(const Array& output) {
    // Find the corresponding output index.
    // If there are more than one array with the same array body in outputs, the first one is always chosen, no matter what array the caller
    // actually specified. It doesn't matter because the array GetRetainedOutput would return is the same.
    // TODO(niboshi): It may be costly in ops with many output arrays.
    auto it = std::find_if(outputs_.begin(), outputs_.end(), [&output](const Array& output2) {
        return internal::GetArrayBody(output) == internal::GetArrayBody(output2);
    });
    if (it == outputs_.end()) {
        throw XchainerError{"Cannot retain an array which is not any of output arrays."};
    }
    size_t output_index = std::distance(outputs_.begin(), it);
    return {internal::GetArrayBody(output)->GetParams(), output_index};
}

}  // namespace xchainer
