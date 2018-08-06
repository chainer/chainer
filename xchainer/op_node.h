#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array_fwd.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"

namespace xchainer {

class BackwardContext;
class Device;

using BackwardFunction = std::function<void(BackwardContext&)>;

namespace internal {

class ArrayNode;
class OpNode;

struct ArrayProps {
    explicit ArrayProps(const Array& array);
    explicit ArrayProps(const ArrayNode& array_node);

    Shape shape;
    Dtype dtype;
    Device& device;
};

class OpNodeBackwardEntry {
public:
    OpNodeBackwardEntry(OpNode& op_node, std::vector<size_t> next_array_node_indices, BackwardFunction backward_func);

    OpNode& op_node() const { return op_node_; }

    size_t next_array_node_count() const { return next_array_node_indices_.size(); }

    const std::vector<size_t>& next_array_node_indices() const { return next_array_node_indices_; }

    // Returns the next array nodes of exotic graphs.
    const std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>>& exotic_next_array_nodes() const {
        return exotic_next_array_nodes_;
    }

    const BackwardFunction& backward_func() const { return backward_func_; }

    void AddExoticNextArrayNode(std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>> next_array_nodes);

private:
    friend class OpNode;

    OpNode& op_node_;

    // The index mapping from local (this backward function) to global (op node).
    // Can be unset if the input array does not require grad.
    std::vector<size_t> next_array_node_indices_;

    std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>> exotic_next_array_nodes_;

    BackwardFunction backward_func_;
};

// Creates a prev array node at the specified index and adds edges between the prev array node and the op node.
// Undefined behavior if the prev array node already exists.
// This function is used by BackwardContext::GetRetainedOutput().
std::shared_ptr<ArrayNode> FabricatePrevArrayNode(std::shared_ptr<OpNode> op_node, size_t prev_array_node_index);

class OpNode {
public:
    // Creates a new op node that has prev array nodes corresponding to the given outputs.
    static std::shared_ptr<OpNode> CreateWithPrevArrayNodes(
            std::string name, BackpropId backprop_id, size_t input_count, const std::vector<ConstArrayRef>& outputs);

    OpNode(const OpNode&) = delete;
    OpNode(OpNode&&) = delete;
    OpNode& operator=(const OpNode&) = delete;
    OpNode& operator=(OpNode&&) = delete;

    OpNodeBackwardEntry& RegisterBackwardFunction(
            std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> next_array_nodes, BackwardFunction backward_func);

    // Adds links to previous array nodes of other graphs.
    void AddEdgesToPreviousArrayNodesOfOuterGraph(
            const BackpropId& outer_backprop_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_prev_array_nodes);

    void Unchain() {
        backward_entries_.clear();
        std::fill(next_array_nodes_.begin(), next_array_nodes_.end(), std::shared_ptr<ArrayNode>{});
        AssertConsistency();
    }

    bool HasNextArrayNode(size_t next_index) const { return next_array_nodes_[next_index] != nullptr; }

    std::string name() const { return name_; }

    std::vector<std::shared_ptr<ArrayNode>>& next_array_nodes();

    const std::vector<std::shared_ptr<ArrayNode>>& next_array_nodes() const;

    gsl::span<OpNodeBackwardEntry> backward_entries() { return backward_entries_; }

    gsl::span<const OpNodeBackwardEntry> backward_entries() const { return backward_entries_; }

    size_t next_array_node_count() const { return next_array_nodes_.size(); }

    size_t prev_array_node_count() const { return prev_array_props_.size(); }

    int64_t rank() const { return rank_; }

    BackpropId backprop_id() const { return backprop_id_; }

    const ArrayProps& GetPrevArrayProps(size_t i) const {
        assert(i < prev_array_props_.size());
        return prev_array_props_[i];
    }

    // Returns the list of prev array nodes on "this" graph.
    const std::vector<std::weak_ptr<ArrayNode>>& prev_array_nodes() const { return prev_array_nodes_; }

    // Returns the list of prev array nodes on "this" graph.
    std::vector<std::weak_ptr<ArrayNode>>& prev_array_nodes() { return prev_array_nodes_; }

    // Returns the previous array nodes of all graphs.
    const std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>>& outer_graphs_prev_array_nodes() const {
        return outer_graphs_prev_array_nodes_;
    }

private:
    OpNode(std::string name, BackpropId backprop_id, size_t next_array_node_count);

    void AssertConsistency() const;

    std::string name_;

    // Backprop ID.
    // Backprop ID is also held in the first entry of prev_array_nodes_, but the reference to it may be invalidated, whereas this member is
    // stable during the lifetime of this OpNode instance.
    BackpropId backprop_id_;

    int64_t rank_{0};

    // List of next array nodes.
    std::vector<std::shared_ptr<ArrayNode>> next_array_nodes_;

    // List of previous array nodes of this graph.
    std::vector<std::weak_ptr<ArrayNode>> prev_array_nodes_;

    // List of prev array nodes of outer graphs.
    // Outer graphs refer to graphs with lower ordinals.
    // Each entry is a pair of backprop ID and list of previous array nodes.
    std::vector<std::tuple<BackpropId, std::vector<std::shared_ptr<ArrayNode>>>> outer_graphs_prev_array_nodes_;

    // Array props of previous array nodes. This is used for creating dummy gradients.
    std::vector<ArrayProps> prev_array_props_;

    std::vector<OpNodeBackwardEntry> backward_entries_;
};

}  // namespace internal
}  // namespace xchainer
