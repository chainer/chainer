#include "xchainer/op_node.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "xchainer/array_node.h"
#include "xchainer/backward.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

OpNodeBackwardEntry::OpNodeBackwardEntry(std::vector<size_t> next_array_node_indices, BackwardFunction backward_func)
    : next_array_node_indices_{std::move(next_array_node_indices)}, backward_func_{std::move(backward_func)} {}

}  // namespace internal

OpNode::OpNode(std::string name, const std::vector<std::shared_ptr<ArrayNode>>& prev_array_nodes)
    : name_{std::move(name)}, graph_id_{prev_array_nodes.front()->graph_id()} {
    // Create weak_ptrs to previous array nodes
    for (const std::shared_ptr<ArrayNode>& prev_array_node : prev_array_nodes) {
        prev_array_nodes_.emplace_back(prev_array_node);
    }
}

gsl::span<std::shared_ptr<ArrayNode>> OpNode::next_nodes() {
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node != nullptr;
    }));
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

gsl::span<const std::shared_ptr<ArrayNode>> OpNode::next_nodes() const {
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node != nullptr;
    }));
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

void OpNode::RegisterBackwardFunction(
        gsl::span<std::reference_wrapper<std::shared_ptr<ArrayNode>>> next_array_nodes, BackwardFunction backward_func) {
    assert(!next_array_nodes.empty());

    // Update the rank of op node
    for (const std::shared_ptr<ArrayNode>& next_array_node : next_array_nodes) {
        rank_ = std::max(rank_, next_array_node->rank());
    }

    // Store next nodes and record indices of them
    std::vector<size_t> next_array_node_indices;
    next_array_node_indices.reserve(next_array_nodes.size());
    for (const std::shared_ptr<ArrayNode>& next_array_node : next_array_nodes) {
        next_array_node_indices.emplace_back(next_array_nodes_.size());
        next_array_nodes_.emplace_back(next_array_node);
    }

    // Add backward entry
    backward_entries_.emplace_back(std::move(next_array_node_indices), std::move(backward_func));
}

}  // namespace xchainer
