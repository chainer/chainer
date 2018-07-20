#include "xchainer/op_node.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "xchainer/array_node.h"
#include "xchainer/backward.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

OpNodeBackwardEntry::OpNodeBackwardEntry(
        OpNode& op_node, std::vector<nonstd::optional<size_t>> next_array_node_indices, BackwardFunction backward_func)
    : op_node_{op_node}, next_array_node_indices_{std::move(next_array_node_indices)}, backward_func_{std::move(backward_func)} {}

std::vector<std::shared_ptr<ArrayNode>> OpNodeBackwardEntry::GetNextArrayNodes() const {
    std::vector<std::shared_ptr<ArrayNode>> array_nodes;
    array_nodes.reserve(next_array_node_indices_.size());
    for (nonstd::optional<size_t> index : next_array_node_indices_) {
        if (index.has_value()) {
            array_nodes.emplace_back(gsl::at(op_node_.next_array_nodes(), *index));
        } else {
            array_nodes.emplace_back(nullptr);
        }
    }
    return array_nodes;
}

void OpNodeBackwardEntry::AddExoticNextArrayNode(std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>> next_array_nodes) {
    assert(std::get<0>(next_array_nodes) != op_node_.graph_id());
    exotic_next_array_nodes_.emplace_back(std::move(next_array_nodes));
}

}  // namespace internal

OpNode::OpNode(
        std::string name,
        GraphId graph_id,
        std::vector<std::weak_ptr<ArrayNode>> prev_array_nodes,
        std::vector<internal::ArrayProps> prev_array_props)
    : name_{std::move(name)},
      graph_id_{graph_id},
      prev_array_nodes_{std::move(prev_array_nodes)},
      prev_array_props_{std::move(prev_array_props)} {
    assert(prev_array_props_.size() == prev_array_nodes_.size());
    AssertConsistency();
}

void OpNode::AssertConsistency() const {
#ifndef NDEBUG
    // No pair of entries may have the same graph ID.
    assert(std::all_of(outer_graphs_prev_array_nodes_.begin(), outer_graphs_prev_array_nodes_.end(), [this](const auto& tup1) {
        return std::all_of(outer_graphs_prev_array_nodes_.begin(), outer_graphs_prev_array_nodes_.end(), [&tup1](const auto& tup2) {
            return &tup1 == &tup2 || std::get<0>(tup1) != std::get<0>(tup2);
        });
    }));

    // Corresponding previous array nodes across graphs (corresponding to the same output array) should have the same array body, if it's
    // alive.
    for (size_t i_prev = 0; i_prev < prev_node_count(); ++i_prev) {
        nonstd::optional<internal::ArrayBody*> prev_array_body{};
        for (const auto& tup : outer_graphs_prev_array_nodes_) {
            const std::vector<std::shared_ptr<ArrayNode>>& vec = std::get<1>(tup);
            const std::shared_ptr<ArrayNode>& prev_array_node = vec[i_prev];
            std::shared_ptr<internal::ArrayBody> body = prev_array_node->GetBody();
            if (!prev_array_body.has_value()) {
                prev_array_body = body.get();
            } else {
                assert(*prev_array_body == body.get());
            }
        }
    }
#endif  // NDEBUG
}

std::vector<std::shared_ptr<ArrayNode>>& OpNode::next_array_nodes() {
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node != nullptr;
    }));
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

const std::vector<std::shared_ptr<ArrayNode>>& OpNode::next_array_nodes() const {
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node != nullptr;
    }));
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

internal::OpNodeBackwardEntry& OpNode::RegisterBackwardFunction(
        std::vector<std::shared_ptr<ArrayNode>> next_array_nodes, BackwardFunction backward_func) {
    AssertConsistency();
    assert(!next_array_nodes.empty());
    assert(std::all_of(next_array_nodes.begin(), next_array_nodes.end(), [this](const std::shared_ptr<ArrayNode>& next_array_node) {
        // next_array_node could be nullptr, if the corresponding input array does not require grad.
        return next_array_node == nullptr || next_array_node->graph_id() == graph_id_;
    }));

    // Update the rank of op node
    for (const std::shared_ptr<ArrayNode>& next_array_node : next_array_nodes) {
        if (next_array_node != nullptr) {
            rank_ = std::max(rank_, next_array_node->rank() + 1);
        }
    }

    // Store next nodes and record indices of them
    std::vector<nonstd::optional<size_t>> next_array_node_indices;
    next_array_node_indices.reserve(next_array_nodes.size());
    for (std::shared_ptr<ArrayNode>& next_array_node : next_array_nodes) {
        if (next_array_node != nullptr) {
            next_array_node_indices.emplace_back(next_array_nodes_.size());
            next_array_nodes_.emplace_back(std::move(next_array_node));
        } else {
            next_array_node_indices.emplace_back(nonstd::nullopt);
        }
    }

    // Add backward entry
    backward_entries_.emplace_back(*this, std::move(next_array_node_indices), std::move(backward_func));

    AssertConsistency();
    return backward_entries_.back();
}

void OpNode::RegisterOuterGraphsPreviousArrayNodes(
        const GraphId& other_graph_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_prev_array_nodes) {
    AssertConsistency();
    assert(other_graph_id != graph_id_);
    assert(outer_graphs_prev_array_nodes.size() == prev_array_props_.size());

    outer_graphs_prev_array_nodes_.emplace_back(other_graph_id, std::move(outer_graphs_prev_array_nodes));

    AssertConsistency();
}

namespace internal {

std::shared_ptr<ArrayNode> CreatePrevArrayNode(std::shared_ptr<OpNode> op_node, size_t prev_array_node_index) {
    assert(prev_array_node_index < op_node->prev_node_count());
    assert(op_node->prev_array_nodes()[prev_array_node_index].expired());

    const internal::ArrayProps& props = op_node->GetPrevArrayProps(prev_array_node_index);
    auto prev_array_node = std::make_shared<ArrayNode>(props.shape, props.dtype, props.device, op_node->graph_id());
    prev_array_node->set_next_op_node(std::move(op_node));
    op_node->prev_array_nodes()[prev_array_node_index] = prev_array_node;

    return prev_array_node;
}

}  // namespace internal
}  // namespace xchainer
