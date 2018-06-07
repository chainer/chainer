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

OpNodeBackwardEntry::OpNodeBackwardEntry(std::vector<size_t> next_node_indices, std::function<void(BackwardContext&)> backward_func)
    : next_node_indices_{std::move(next_node_indices)}, backward_func_{std::move(backward_func)} {}

}  // namespace internal

OpNode::OpNode(std::string name, const std::vector<std::shared_ptr<ArrayNode>>& prev_nodes)
    : name_{std::move(name)}, graph_id_{prev_nodes.front()->graph_id()} {
    // Create weak_ptrs to previous array nodes
    for (const std::shared_ptr<ArrayNode>& prev_node : prev_nodes) {
        prev_nodes_.emplace_back(prev_node);
    }
}

void OpNode::RegisterBackwardFunction(
        gsl::span<std::reference_wrapper<std::shared_ptr<ArrayNode>>> next_nodes, std::function<void(BackwardContext&)>&& backward_func) {
    // Set this to a large enough number to avoid copy of shared pointers
    static constexpr size_t kDefaultNextNodesReserveCount = 5U;

    assert(!next_nodes.empty());

    // Update the rank of op node
    for (const std::shared_ptr<ArrayNode>& next_node : next_nodes) {
        rank_ = std::max(rank_, next_node->rank());
    }

    // Store next nodes and record indices of them
    std::vector<size_t> next_node_indices;
    // Reserve
    next_node_indices.reserve(next_nodes.size());
    if (next_nodes_.empty()) {
        next_nodes_.reserve(kDefaultNextNodesReserveCount);
    } else {
        next_nodes_.reserve(next_nodes_.size() + next_nodes.size());
    }
    // Store
    for (const std::shared_ptr<ArrayNode>& next_node : next_nodes) {
        next_node_indices.emplace_back(next_nodes_.size());
        next_nodes_.emplace_back(next_node);
    }

    // Add backward entry
    if (backward_entries_.empty()) {
        backward_entries_.reserve(kDefaultNextNodesReserveCount);
    }
    backward_entries_.emplace_back(std::move(next_node_indices), std::move(backward_func));
}

}  // namespace xchainer
