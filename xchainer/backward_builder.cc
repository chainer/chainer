#include "xchainer/backward_builder.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/device.h"
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
            if (!IsBackpropRequired(graph_id)) {
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

        std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> temp_next_array_nodes;
        temp_next_array_nodes.reserve(next_array_nodes.size());
        std::transform(
                input_indices_.begin(),
                input_indices_.end(),
                std::back_inserter(temp_next_array_nodes),
                [&next_array_nodes](size_t input_index) {
                    const std::shared_ptr<ArrayNode>* array_node = next_array_nodes[input_index];
                    return std::make_tuple(input_index, array_node == nullptr ? nullptr : *array_node);
                });

        std::shared_ptr<OpNode>& op_node = builder_.FindOrCreateOpNode(graph_id);
        op_node->RegisterBackwardFunction(std::move(temp_next_array_nodes), backward_func);
    }
}

BackwardBuilder::BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, std::vector<ConstArrayRef> outputs)
    : op_name_{op_name},
      inputs_{std::move(inputs)},
      inputs_target_created_(inputs_.size()),
      outputs_{std::move(outputs)},
      input_retention_record_{inputs_.size()},
      output_retention_record_{outputs_.size()} {
    assert(!inputs_.empty());
    assert(!outputs_.empty());
    assert(inputs_.size() == inputs_target_created_.size());
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
    }

    assert(!op_node_map_.empty());
    return insert_result.first->second;
}

RetainedInputToken BackwardBuilder::RetainInput(size_t input_index) {
    assert(input_index < inputs_.size());
    input_retention_record_.Record(input_index);
    return {internal::GetArrayBody(gsl::at(inputs_, input_index))->GetParams(), input_index};
}

RetainedOutputToken BackwardBuilder::RetainOutput(size_t output_index) {
    assert(output_index < outputs_.size());
    output_retention_record_.Record(output_index);
    return {internal::GetArrayBody(gsl::at(outputs_, output_index))->GetParams(), output_index};
}

void BackwardBuilder::Finalize() {
    assert(!is_finalized_);
    // Checks that the backward definitions cover all the input arrays.
    assert(std::all_of(inputs_target_created_.begin(), inputs_target_created_.end(), [](bool done) { return done; }));

    AddEdgesToArrayNodesBetweenRetainedOuterGraphs();

    is_finalized_ = true;
}

namespace {

void AddEdgesToNextArrayNodesOfOuterGraph(
        const OpNode& outer_op_node,
        OpNode& inner_op_node,
        const backward_builder_detail::RetentionRecord& input_retention_record,
        std::vector<std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>>& input_retention_cache) {
    auto it = std::find_if(
            input_retention_cache.begin(),
            input_retention_cache.end(),
            [&outer_op_node](const std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>& cache) {
                return std::get<0>(cache) == outer_op_node.graph_id();
            });
    if (it != input_retention_cache.end()) {
        inner_op_node.AddEdgesToNextArrayNodesOfOuterGraph(outer_op_node.graph_id(), std::get<1>(*it));
    } else {
        std::vector<std::shared_ptr<internal::ArrayNode>> next_array_nodes;
        next_array_nodes.reserve(input_retention_record.size());

        for (size_t i = 0; i < input_retention_record.size(); ++i) {
            if (input_retention_record.IsRecorded(i)) {
                next_array_nodes.emplace_back(outer_op_node.next_array_nodes()[i]);
            } else {
                next_array_nodes.emplace_back(nullptr);
            }
        }

        input_retention_cache.emplace_back(outer_op_node.graph_id(), next_array_nodes);
        inner_op_node.AddEdgesToNextArrayNodesOfOuterGraph(outer_op_node.graph_id(), std::move(next_array_nodes));
    }
}

void AddEdgesToPreviousArrayNodesOfOuterGraph(
        const OpNode& outer_op_node,
        OpNode& inner_op_node,
        const backward_builder_detail::RetentionRecord& output_retention_record,
        std::vector<std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>>& output_retention_cache) {
    auto it = std::find_if(
            output_retention_cache.begin(),
            output_retention_cache.end(),
            [&outer_op_node](const std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>& cache) {
                return std::get<0>(cache) == outer_op_node.graph_id();
            });
    if (it != output_retention_cache.end()) {
        inner_op_node.AddEdgesToPreviousArrayNodesOfOuterGraph(outer_op_node.graph_id(), std::get<1>(*it));
    } else {
        std::vector<std::shared_ptr<internal::ArrayNode>> prev_array_nodes;
        prev_array_nodes.reserve(output_retention_record.size());

        // Outer graphs must be registered using shared_ptr but op nodes only have weak_ptr to their previous array nodes.
        // Therefore, first convert the weak_ptr to shared_ptr, assuming that they have not expired.
        for (size_t i = 0; i < output_retention_record.size(); ++i) {
            if (output_retention_record.IsRecorded(i)) {
                const std::weak_ptr<ArrayNode>& prev = outer_op_node.prev_array_nodes()[i];
                assert(!prev.expired());
                prev_array_nodes.emplace_back(prev.lock());
            } else {
                prev_array_nodes.emplace_back(nullptr);
            }
        }

        output_retention_cache.emplace_back(outer_op_node.graph_id(), prev_array_nodes);
        inner_op_node.AddEdgesToPreviousArrayNodesOfOuterGraph(outer_op_node.graph_id(), std::move(prev_array_nodes));
    }
}

}  // namespace

void BackwardBuilder::AddEdgesToArrayNodesBetweenRetainedOuterGraphs() {
    if (input_retention_record_.IsAnyRecorded()) {
        // Create a set of graphs to which the retained inputs belong.
        std::unordered_set<GraphId> retained_graphs{};
        for (size_t i = 0; i < input_retention_record_.size(); ++i) {
            if (input_retention_record_.IsRecorded(i)) {
                for (const std::shared_ptr<ArrayNode>& array_node : internal::GetArrayBody(gsl::at(inputs_, i))->nodes()) {
                    retained_graphs.emplace(array_node->graph_id());
                }
            }
        }

        std::vector<std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>> input_retention_cache{};
        for (auto it = retained_graphs.begin(); it != retained_graphs.end(); ++it) {
            const GraphId& graph_id = *it;
            for (auto other_it = retained_graphs.begin(); other_it != retained_graphs.end(); ++other_it) {
                const GraphId& other_graph_id = *other_it;
                if (graph_id < other_graph_id) {
                    AddEdgesToNextArrayNodesOfOuterGraph(
                            *op_node_map_[graph_id], *op_node_map_[other_graph_id], input_retention_record_, input_retention_cache);
                }
            }
        }
    }
    if (output_retention_record_.IsAnyRecorded()) {
        std::vector<std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>> output_retention_cache{};
        for (const auto& tup : op_node_map_) {
            for (const auto& other_tup : op_node_map_) {
                if (tup.first < other_tup.first) {
                    AddEdgesToPreviousArrayNodesOfOuterGraph(
                            *tup.second, *other_tup.second, output_retention_record_, output_retention_cache);
                }
            }
        }
    }
}

}  // namespace xchainer
