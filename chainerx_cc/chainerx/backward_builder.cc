#include "chainerx/backward_builder.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/array_node.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/device.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/op_node.h"

namespace chainerx {
namespace {

using internal::ArrayNode;
using internal::OpNode;

}  // namespace

BackwardBuilder::Target::Target(BackwardBuilder& builder, std::vector<size_t> input_indices)
    : builder_{builder}, input_indices_{std::move(input_indices)} {
    // All input arrays must have the same device.
    CHAINERX_ASSERT(std::all_of(input_indices.begin(), input_indices.end(), [this](size_t input_index) {
        return &gsl::at(builder_.inputs_, input_index).get().device() == &(builder_.inputs_.front().get().device());
    }));

    graph_to_input_array_nodes_ = CreateInputArrayNodesMap();
}

absl::flat_hash_map<BackpropId, BackwardBuilder::Target::InputArrayNodes> BackwardBuilder::Target::CreateInputArrayNodesMap() const {
    absl::flat_hash_map<BackpropId, InputArrayNodes> graph_to_input_array_nodes{};

    if (builder_.has_any_applicable_outputs_) {
        // At least one output arrays can have gradients.

        for (size_t input_index : input_indices_) {
            // Need to access the input array via the builder.
            const Array& input = gsl::at(builder_.inputs_, input_index);

            for (std::shared_ptr<ArrayNode>& input_array_node : internal::GetArrayBody(input)->nodes()) {
                const BackpropId& backprop_id = input_array_node->backprop_id();
                if (!IsBackpropRequired(backprop_id)) {
                    continue;
                }

                // Add the array node to the mapping
                auto insert_result = graph_to_input_array_nodes.emplace(backprop_id, InputArrayNodes{});
                auto& input_array_nodes = insert_result.first->second;
                if (insert_result.second) {
                    // New array node for a graph. Fill all array nodes with nullptr.
                    input_array_nodes.resize(builder_.inputs_.size());
                }
                // Assign valid pointer to the array node.
                input_array_nodes[input_index] = &input_array_node;
            }
        }

        if (CHAINERX_DEBUG) {
            for (auto& pair : graph_to_input_array_nodes) {
                const BackpropId& backprop_id = pair.first;
                (void)backprop_id;  // maybe unused
                const InputArrayNodes& input_array_nodes = pair.second;
                for (const std::shared_ptr<ArrayNode>* array_node : input_array_nodes) {
                    CHAINERX_ASSERT(array_node == nullptr || backprop_id == (*array_node)->backprop_id());
                }
            }
        }
    }

    return graph_to_input_array_nodes;
}

void BackwardBuilder::Target::Define(const BackwardFunction& backward_func) {
    CHAINERX_ASSERT(is_definition_required());

    // Find/Create an op node for each graph and register the given backward function to each of them.
    for (const auto& pair : graph_to_input_array_nodes_) {
        const BackpropId& backprop_id = pair.first;
        const InputArrayNodes& input_array_nodes = pair.second;

        std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> temp_input_array_nodes;
        temp_input_array_nodes.reserve(input_array_nodes.size());
        std::transform(
                input_indices_.begin(),
                input_indices_.end(),
                std::back_inserter(temp_input_array_nodes),
                [&input_array_nodes](size_t input_index) {
                    const std::shared_ptr<ArrayNode>* array_node = input_array_nodes[input_index];
                    return std::make_tuple(input_index, array_node == nullptr ? nullptr : *array_node);
                });

        std::shared_ptr<OpNode>& op_node = builder_.FindOrCreateOpNode(backprop_id);
        op_node->RegisterBackwardFunction(std::move(temp_input_array_nodes), backward_func);
    }
}

BackwardBuilder::BackwardBuilder(const char* op_name, std::vector<ConstArrayRef> inputs, std::vector<ConstArrayRef> outputs)
    : op_name_{op_name},
      context_{inputs.front().get().context()},
      inputs_{std::move(inputs)},
      inputs_target_created_(inputs_.size()),
      outputs_{std::move(outputs)},
      input_retention_record_{inputs_.size()},
      output_retention_record_{outputs_.size()} {
    CHAINERX_ASSERT(!inputs_.empty());
    CHAINERX_ASSERT(!outputs_.empty());
    CHAINERX_ASSERT(inputs_.size() == inputs_target_created_.size());
    CHAINERX_ASSERT(
            std::all_of(inputs_.begin(), inputs_.end(), [](const Array& input) { return internal::GetArrayBody(input) != nullptr; }));
    CHAINERX_ASSERT(
            std::all_of(outputs_.begin(), outputs_.end(), [](const Array& output) { return internal::GetArrayBody(output) != nullptr; }));
    // Outputs requiring grad (e.g. in-place ops.) must have been detected and reported before reaching here.
    CHAINERX_ASSERT(std::all_of(
            outputs_.begin(), outputs_.end(), [](const Array& output) { return internal::GetArrayBody(output)->nodes().empty(); }));
    // Arrays must be on the same device within inputs / outputs respectively.
    CHAINERX_ASSERT(std::all_of(outputs_.begin(), outputs_.end(), [this](const Array& output) {
        return &outputs_.begin()->get().device() == &output.device();
    }));
    CHAINERX_ASSERT(std::all_of(
            inputs_.begin(), inputs_.end(), [this](const Array& input) { return &inputs_.begin()->get().device() == &input.device(); }));

    has_any_applicable_outputs_ =
            std::any_of(outputs_.begin(), outputs_.end(), [](const Array& output) { return GetKind(output.dtype()) == DtypeKind::kFloat; });
}

std::shared_ptr<OpNode>& BackwardBuilder::FindOrCreateOpNode(const BackpropId& backprop_id) {
    // Try to find an existing op node for the given graph.
    auto insert_result = op_node_map_.emplace(backprop_id, nullptr);

    // If not found, create a new one.
    if (insert_result.second) {
        insert_result.first->second = OpNode::CreateWithOutputArrayNodes(op_name_, backprop_id, inputs_.size(), outputs_);
    }

    CHAINERX_ASSERT(!op_node_map_.empty());
    return insert_result.first->second;
}

RetainedInputToken BackwardBuilder::RetainInput(size_t input_index) {
    CHAINERX_ASSERT(input_index < inputs_.size());
    input_retention_record_.Record(input_index);
    return {internal::GetArrayBody(gsl::at(inputs_, input_index))->GetParams(), input_index};
}

std::vector<RetainedInputToken> BackwardBuilder::RetainInput(std::vector<size_t> indices) {
    std::vector<RetainedInputToken> token;
    for (size_t i : indices) {
        CHAINERX_ASSERT(i < inputs_.size());
        input_retention_record_.Record(i);
        token.emplace_back(internal::GetArrayBody(gsl::at(inputs_, i))->GetParams(), i);
    }
    return token;
}

RetainedOutputToken BackwardBuilder::RetainOutput(size_t output_index) {
    CHAINERX_ASSERT(output_index < outputs_.size());
    output_retention_record_.Record(output_index);
    return {internal::GetArrayBody(gsl::at(outputs_, output_index))->GetParams(), output_index};
}

std::vector<RetainedOutputToken> BackwardBuilder::RetainOutput(std::vector<size_t> indices) {
    std::vector<RetainedOutputToken> token;
    for (size_t i : indices) {
        CHAINERX_ASSERT(i < outputs_.size());
        output_retention_record_.Record(i);
        token.emplace_back(internal::GetArrayBody(gsl::at(outputs_, i))->GetParams(), i);
    }
    return token;
}

void BackwardBuilder::Finalize() {
    CHAINERX_ASSERT(!is_finalized_);
    // Checks that the backward definitions cover all the input arrays.
    CHAINERX_ASSERT(std::all_of(inputs_target_created_.begin(), inputs_target_created_.end(), [](bool done) { return done; }));

    AddEdgesFromOpNodeToArrayNodeOfOuterGraphsForRetention();

    // Connect each pair of backprop IDs concerned in this op.
    // If two backprop IDs are connected, backpropping on the one with lower ordinal will prohibit future backprop on the other.
    ConnectBackpropIds();

    is_finalized_ = true;
}

namespace {

void AddEdgesFromOpNodeToInputArrayNodesOfOuterGraph(
        const OpNode& outer_op_node, OpNode& inner_op_node, const backward_builder_detail::RetentionRecord& input_retention_record) {
    std::vector<std::shared_ptr<internal::ArrayNode>> input_array_nodes;
    input_array_nodes.reserve(input_retention_record.size());

    for (size_t i = 0; i < input_retention_record.size(); ++i) {
        if (input_retention_record.IsRecorded(i)) {
            input_array_nodes.emplace_back(outer_op_node.input_array_nodes()[i]);
        } else {
            input_array_nodes.emplace_back(nullptr);
        }
    }

    inner_op_node.AddEdgesToInputArrayNodesOfOuterGraph(outer_op_node.backprop_id(), std::move(input_array_nodes));
}

void AddEdgesFromOpNodeToOutputArrayNodesOfOuterGraph(
        const OpNode& outer_op_node, OpNode& inner_op_node, const backward_builder_detail::RetentionRecord& output_retention_record) {
    std::vector<std::shared_ptr<internal::ArrayNode>> output_array_nodes;
    output_array_nodes.reserve(output_retention_record.size());

    // Outer graphs must be registered using shared_ptr but op nodes only have weak_ptr to their output array nodes.
    // Therefore, first convert the weak_ptr to shared_ptr, assuming that they have not expired.
    for (size_t i = 0; i < output_retention_record.size(); ++i) {
        if (output_retention_record.IsRecorded(i)) {
            const absl::optional<std::weak_ptr<ArrayNode>>& array_node = outer_op_node.output_array_nodes()[i];
            CHAINERX_ASSERT(array_node.has_value());
            CHAINERX_ASSERT(!array_node->expired());
            output_array_nodes.emplace_back(array_node->lock());
        } else {
            output_array_nodes.emplace_back(nullptr);
        }
    }

    inner_op_node.AddEdgesToOutputArrayNodesOfOuterGraph(outer_op_node.backprop_id(), std::move(output_array_nodes));
}

}  // namespace

void BackwardBuilder::AddEdgesFromOpNodeToArrayNodeOfOuterGraphsForRetention() {
    // Create edges from op nodes to outer graph array nodes so that retained inputs and output can be restored.
    // For outputs, we need to consider all graphs that this builder defines since each output participates in all of them.
    // For inputs, we only need to consider a subset of the graphs; the graphs that each input belongs to.

    // Add edges to input array nodes
    if (input_retention_record_.IsAnyRecorded()) {
        // Collect graphs to which the retained inputs belong.
        // TODO(beam2d): Use a lighter container.
        std::unordered_set<BackpropId> retained_graphs{};
        for (size_t i = 0; i < input_retention_record_.size(); ++i) {
            if (input_retention_record_.IsRecorded(i)) {
                for (const std::shared_ptr<ArrayNode>& array_node : internal::GetArrayBody(gsl::at(inputs_, i))->nodes()) {
                    retained_graphs.emplace(array_node->backprop_id());
                }
            }
        }

        // Add edges to the input array nodes belonging to the collected graphs.
        for (const BackpropId& backprop_id : retained_graphs) {
            const OpNode& op_node = *op_node_map_.at(backprop_id);
            for (const BackpropId& other_backprop_id : retained_graphs) {
                if (backprop_id < other_backprop_id) {
                    OpNode& other_op_node = *op_node_map_.at(other_backprop_id);
                    AddEdgesFromOpNodeToInputArrayNodesOfOuterGraph(op_node, other_op_node, input_retention_record_);
                }
            }
        }
    }

    // Add edges to output array nodes
    if (output_retention_record_.IsAnyRecorded()) {
        for (const auto& tup : op_node_map_) {
            const BackpropId& backprop_id = tup.first;
            const OpNode& op_node = *tup.second;
            for (const auto& other_tup : op_node_map_) {
                const BackpropId& other_backprop_id = other_tup.first;
                OpNode& other_op_node = *other_tup.second;
                if (backprop_id < other_backprop_id) {
                    AddEdgesFromOpNodeToOutputArrayNodesOfOuterGraph(op_node, other_op_node, output_retention_record_);
                }
            }
        }
    }
}

void BackwardBuilder::ConnectBackpropIds() {
    for (auto it1 = op_node_map_.begin(); it1 != op_node_map_.end(); ++it1) {
        const BackpropId& backprop_id1 = it1->first;
        for (auto it2 = std::next(it1); it2 != op_node_map_.end(); ++it2) {
            const BackpropId& backprop_id2 = it2->first;
            context_.ConnectBackpropIds(backprop_id1, backprop_id2);
        }
    }
}

}  // namespace chainerx
