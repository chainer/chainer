#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.h>
#include <absl/types/span.h>

#include "chainerx/array_body.h"
#include "chainerx/array_fwd.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"

namespace chainerx {

class BackwardContext;
class Device;

using BackwardFunction = std::function<void(BackwardContext&)>;

namespace internal {

class ArrayNode;
class OpNode;

struct ArrayProps {
    explicit ArrayProps(const Array& array);
    explicit ArrayProps(const ArrayNode& array_node);
    explicit ArrayProps(const ArrayBody& array_body);

    Shape shape;
    Dtype dtype;
    Device& device;
};

class OpNodeBackwardEntry {
public:
    OpNodeBackwardEntry(OpNode& op_node, std::vector<size_t> input_array_node_indices, BackwardFunction backward_func);

    OpNode& op_node() const { return op_node_; }

    size_t input_array_node_count() const { return input_array_node_indices_.size(); }

    const std::vector<size_t>& input_array_node_indices() const { return input_array_node_indices_; }

    const BackwardFunction& backward_func() const { return backward_func_; }

private:
    friend class OpNode;

    OpNode& op_node_;

    // The index mapping from local (this backward function) to global (op node).
    // Can be unset if the input array does not require grad.
    std::vector<size_t> input_array_node_indices_;

    BackwardFunction backward_func_;
};

// Creates an output array node at the specified index and adds edges between the output array node and the op node.
// Undefined behavior if the output array node already exists.
// This function is used by BackwardContext::GetRetainedOutput().
std::shared_ptr<ArrayNode> FabricateOutputArrayNode(std::shared_ptr<OpNode> op_node, size_t output_array_node_index);

class OpNode {
public:
    // Creates a new op node that has output array nodes corresponding to the given outputs.
    static std::shared_ptr<OpNode> CreateWithOutputArrayNodes(
            std::string name, BackpropId backprop_id, size_t input_count, const std::vector<ConstArrayRef>& outputs);

    ~OpNode() = default;

    OpNode(const OpNode&) = delete;
    OpNode(OpNode&&) = delete;
    OpNode& operator=(const OpNode&) = delete;
    OpNode& operator=(OpNode&&) = delete;

    OpNodeBackwardEntry& RegisterBackwardFunction(
            std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> input_array_nodes, BackwardFunction backward_func);

    // Adds links to input array nodes of other graphs.
    // The size of the vector must be equal to the number of inputs.
    void AddEdgesToInputArrayNodesOfOuterGraph(
            const BackpropId& outer_backprop_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_input_array_nodes);

    // Adds links to output array nodes of other graphs.
    // The size of the vector must be equal to the number of outputs.
    void AddEdgesToOutputArrayNodesOfOuterGraph(
            const BackpropId& outer_backprop_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_output_array_nodes);

    void Unchain() {
        backward_entries_.clear();
        std::fill(input_array_nodes_.begin(), input_array_nodes_.end(), std::shared_ptr<ArrayNode>{});
        AssertConsistency();
    }

    bool HasInputArrayNode(size_t input_index) const { return input_array_nodes_[input_index] != nullptr; }

    std::string name() const { return name_; }

    std::vector<std::shared_ptr<ArrayNode>>& input_array_nodes();

    const std::vector<std::shared_ptr<ArrayNode>>& input_array_nodes() const;

    absl::Span<OpNodeBackwardEntry> backward_entries() { return absl::MakeSpan(backward_entries_); }

    absl::Span<const OpNodeBackwardEntry> backward_entries() const { return absl::MakeConstSpan(backward_entries_); }

    size_t input_array_node_count() const { return input_array_nodes_.size(); }

    size_t output_array_node_count() const { return output_array_props_.size(); }

    int64_t rank() const { return rank_; }

    BackpropId backprop_id() const { return backprop_id_; }

    const ArrayProps& GetOutputArrayProps(size_t i) const {
        CHAINERX_ASSERT(i < output_array_props_.size());
        return output_array_props_[i];
    }

    // Returns the list of output array nodes on "this" graph.
    const std::vector<absl::optional<std::weak_ptr<ArrayNode>>>& output_array_nodes() const { return output_array_nodes_; }

    // Returns the list of output array nodes on "this" graph.
    std::vector<absl::optional<std::weak_ptr<ArrayNode>>>& output_array_nodes() { return output_array_nodes_; }

    // Returns the input array nodes of all graphs.
    const std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>>& outer_graphs_input_array_nodes() const {
        return outer_graphs_input_array_nodes_;
    }

    // Returns the output array nodes of all graphs.
    const std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>>& outer_graphs_output_array_nodes() const {
        return outer_graphs_output_array_nodes_;
    }

private:
    OpNode(std::string name, BackpropId backprop_id, size_t input_array_node_count);

    void AssertConsistency() const;

    std::string name_;

    // Backprop ID.
    // Backprop ID is also held in the first entry of output_array_nodes_, but the reference to it may be invalidated, whereas this member
    // is stable during the lifetime of this OpNode instance.
    BackpropId backprop_id_;

    int64_t rank_{0};

    // List of input array nodes.
    std::vector<std::shared_ptr<ArrayNode>> input_array_nodes_;

    // List of output array nodes of this graph.
    std::vector<absl::optional<std::weak_ptr<ArrayNode>>> output_array_nodes_;

    // List of input/output array nodes of outer graphs.
    // Outer graphs refer to graphs with lower ordinals.
    // Each entry is a pair of backprop ID and list of input/output array nodes.
    std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>> outer_graphs_input_array_nodes_;
    std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>> outer_graphs_output_array_nodes_;

    // Array props of output array nodes. This is used for creating dummy gradients.
    std::vector<ArrayProps> output_array_props_;

    std::vector<OpNodeBackwardEntry> backward_entries_;
};

}  // namespace internal
}  // namespace chainerx
