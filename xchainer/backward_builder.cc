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

BackwardBuilder::Target::Target(BackwardBuilder& builder, std::vector<size_t> input_indices)
    : builder_{builder}, input_indices_{std::move(input_indices)} {
    // All input arrays must have the same device.
    assert(std::all_of(input_indices.begin(), input_indices.end(), [this](size_t input_index) {
        return &gsl::at(builder_.inputs_, input_index).get().device() == &(builder_.inputs_.front().get().device());
    }));

    KeepGraphsAndArrayNodesThatRequireDefinition();
}

void BackwardBuilder::Target::KeepGraphsAndArrayNodesThatRequireDefinition() {
    assert(graph_to_next_array_nodes_.empty());
    for (size_t input_index : input_indices_) {
        // Need to access the input array via the builder.
        const Array& input = gsl::at(builder_.inputs_, input_index);

        for (std::shared_ptr<ArrayNode>& next_array_node : internal::GetArrayBody(input)->nodes()) {
            const GraphId& graph_id = next_array_node->graph_id();

            if (!IsBackpropRequired(graph_id, input.device().context())) {
                continue;
            }

            // Add the array node to the mapping
            auto insert_result = graph_to_next_array_nodes_.emplace(graph_id, NextArrayNodes{});
            auto& next_array_nodes = insert_result.first->second;
            if (insert_result.second) {
                // New array node for a graph. Fill all array nodes with nullptr.
                next_array_nodes.resize(builder_.inputs_.size());
            }
            // Assign valid pointer to the array node.
            next_array_nodes[input_index] = &next_array_node;
        }
    }

#ifndef NDEBUG
    for (auto& pair : graph_to_next_array_nodes_) {
        const GraphId& graph_id = pair.first;
        const NextArrayNodes& next_array_nodes = pair.second;
        for (const std::shared_ptr<ArrayNode>* array_node : next_array_nodes) {
            assert(array_node == nullptr || graph_id == (*array_node)->graph_id());
        }
    }
#endif  // NDEBUG
}

void BackwardBuilder::Target::Define(const BackwardFunction& backward_func) {
    assert(is_definition_required());

    // Find/Create an op node for each graph and register the given backward function to each of them.
    for (const auto& pair : graph_to_next_array_nodes_) {
        const GraphId& graph_id = pair.first;
        const NextArrayNodes& next_array_nodes = pair.second;

        std::shared_ptr<OpNode>& op_node = builder_.FindOrCreateOpNode(graph_id);

        std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> temp_next_array_nodes;
        temp_next_array_nodes.reserve(next_array_nodes.size());
        std::transform(
                input_indices_.begin(),
                input_indices_.end(),
                std::back_inserter(temp_next_array_nodes),
                [&next_array_nodes](size_t input_index) {
                    const std::shared_ptr<ArrayNode>* array_node = next_array_nodes[input_index];
                    return std::tuple<size_t, std::shared_ptr<ArrayNode>>{input_index, array_node == nullptr ? nullptr : *array_node};
                });

        op_node->RegisterBackwardFunction(std::move(temp_next_array_nodes), backward_func);
    }
}

BackwardBuilder::BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, std::vector<ConstArrayRef> outputs)
    : op_name_{op_name}, inputs_{std::move(inputs)}, outputs_{std::move(outputs)} {
    // Outputs requiring grad (e.g. in-place ops.) must have been detected and reported before reaching here.
    assert(std::all_of(
            outputs_.begin(), outputs_.end(), [](const Array& output) { return internal::GetArrayBody(output)->nodes().empty(); }));
    // Arrays must be on the same device within inputs / outputs respectively.
    assert(std::all_of(outputs_.begin(), outputs_.end(), [this](const Array& output) {
        return &outputs_.begin()->get().device() == &output.device();
    }));
    assert(std::all_of(
            inputs_.begin(), inputs_.end(), [this](const Array& input) { return &inputs_.begin()->get().device() == &input.device(); }));
}

std::shared_ptr<OpNode>& BackwardBuilder::FindOrCreateOpNode(const GraphId& graph_id) {
    // Try to find an existing op node for the given graph.
    auto insert_result = op_node_map_.emplace(graph_id, nullptr);

    // If not found, create a new one.
    if (insert_result.second) {
        insert_result.first->second = OpNode::CreateWithPrevArrayNodes(op_name_, graph_id, inputs_.size(), outputs_);
        RegisterOuterGraphsPreviousArrayNodes(insert_result.first->second);
    }

    assert(!op_node_map_.empty());
    return insert_result.first->second;
}

void BackwardBuilder::RegisterOuterGraphsPreviousArrayNodes(const std::shared_ptr<internal::OpNode>& op_node) {
    const GraphId& graph_id = op_node->graph_id();

    // Lazily initialize graph to prev array node mappings.
    if (graph_to_prev_array_nodes_.empty()) {
        for (const Array& output : outputs_) {
            for (std::shared_ptr<ArrayNode> output_array_node : internal::GetArrayBody(output)->nodes()) {
                graph_to_prev_array_nodes_[output_array_node->graph_id()].emplace_back(std::move(output_array_node));
            }
        }
    }

    // Compare the order (outer/inner) between the graph of given op node and all graphs involved in this builder.
    // Then create references to outer graphs as necessary.
    for (const auto& tup : graph_to_prev_array_nodes_) {
        const GraphId& other_graph_id = tup.first;

        if (other_graph_id < graph_id) {
            // Create reference from given (inner) to other (outer).
            const std::vector<std::shared_ptr<internal::ArrayNode>>& prev_array_nodes = tup.second;
            assert(prev_array_nodes.size() == outputs_.size());
            op_node->RegisterOuterGraphsPreviousArrayNodes(other_graph_id, prev_array_nodes);
        } else if (other_graph_id > graph_id) {
            // Create reference from other (inner) to given (outer).
            const std::vector<std::weak_ptr<ArrayNode>>& prev_array_nodes = op_node->prev_array_nodes();
            assert(prev_array_nodes.size() == outputs_.size());
            std::vector<std::shared_ptr<internal::ArrayNode>> temp_prev_array_nodes;
            temp_prev_array_nodes.reserve(prev_array_nodes.size());
            std::transform(
                    prev_array_nodes.begin(),
                    prev_array_nodes.end(),
                    std::back_inserter(temp_prev_array_nodes),
                    [](const std::weak_ptr<ArrayNode>& prev) {
                        assert(!prev.expired());
                        return prev.lock();
                    });
            op_node_map_[other_graph_id]->RegisterOuterGraphsPreviousArrayNodes(graph_id, std::move(temp_prev_array_nodes));
        } else {
            // Do nothing.
        }
    }
}

RetainedInputToken BackwardBuilder::RetainInput(size_t input_index) {
    assert(input_index < inputs_.size());
    return {internal::GetArrayBody(gsl::at(inputs_, input_index))->GetParams(), input_index};
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
