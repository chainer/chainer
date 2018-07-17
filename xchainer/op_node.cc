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
      graph_id_{std::move(graph_id)},
      prev_array_nodes_{std::make_tuple(graph_id_, std::move(prev_array_nodes))},
      prev_array_props_{std::move(prev_array_props)} {
    assert(prev_array_props_.size() == std::get<1>(prev_array_nodes_[0]).size());
    AssertConsistency();
}

void OpNode::AssertConsistency() const {
#ifndef NDEBUG
    assert(!prev_array_nodes_.empty());

    // The first entry corresponds to "this" graph.
    assert(std::get<0>(prev_array_nodes_[0]) == graph_id_);

    // No pair of entries may have the same graph ID.
    assert(std::all_of(prev_array_nodes_.begin(), prev_array_nodes_.end(), [this](const auto& tup1) {
        return std::all_of(prev_array_nodes_.begin(), prev_array_nodes_.end(), [&tup1](const auto& tup2) {
            return &tup1 == &tup2 || std::get<0>(tup1) != std::get<0>(tup2);
        });
    }));

    // Corresonding previous array nodes across graphs (corresponding to the same output array) should have the same array body, if it's
    // alive.
    size_t n_prev = std::get<1>(prev_array_nodes_[0]).size();
    for (size_t i_prev = 0; i_prev < n_prev; ++i_prev) {
        nonstd::optional<internal::ArrayBody*> prev_array_body{};
        for (const auto& tup : prev_array_nodes_) {
            const std::vector<std::weak_ptr<ArrayNode>>& vec = std::get<1>(tup);
            if (std::shared_ptr<ArrayNode> prev_array_node = vec[i_prev].lock()) {
                std::shared_ptr<internal::ArrayBody> body = prev_array_node->GetBody();
                if (!prev_array_body.has_value()) {
                    prev_array_body = body.get();
                } else {
                    assert(*prev_array_body == body.get());
                }
            }
        }
    }
#endif  // NDEBUG
}

gsl::span<std::shared_ptr<ArrayNode>> OpNode::next_array_nodes() {
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node != nullptr;
    }));
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

gsl::span<const std::shared_ptr<ArrayNode>> OpNode::next_array_nodes() const {
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

void OpNode::RegisterExoticPreviousArrayNodes(GraphId other_graph_id, std::vector<std::weak_ptr<ArrayNode>> exotic_prev_array_nodes) {
    AssertConsistency();
    assert(other_graph_id != graph_id_);
    assert(exotic_prev_array_nodes.size() == prev_array_props_.size());

    prev_array_nodes_.emplace_back(std::move(other_graph_id), std::move(exotic_prev_array_nodes));

    AssertConsistency();
}

std::shared_ptr<OpNode> OpNode::CloneInOtherGraph(const GraphId& other_graph_id) const {
    AssertConsistency();
    assert(graph_id_ != other_graph_id);

    const std::vector<std::weak_ptr<ArrayNode>>* new_prev_array_nodes{};
    std::vector<std::tuple<GraphId, std::vector<std::weak_ptr<ArrayNode>>>> new_exotic_prev_array_nodes;

    // Copy prev array nodes
    for (const std::tuple<GraphId, std::vector<std::weak_ptr<ArrayNode>>>& pair : prev_array_nodes_) {
        const GraphId& graph_id = std::get<0>(pair);
        const std::vector<std::weak_ptr<ArrayNode>>& vec = std::get<1>(pair);
        assert(vec.size() == prev_array_props_.size());

        if (graph_id == other_graph_id) {
            // Prev array nodes for new "this" graph.
            new_prev_array_nodes = &vec;
        } else {
            // Prev array nodes for new "exotic" graph.
            new_exotic_prev_array_nodes.emplace_back(graph_id, vec);
        }
    }
    assert(new_prev_array_nodes != nullptr);

    // Create new op node instance
    auto op_node = std::make_shared<OpNode>(name_, other_graph_id, *new_prev_array_nodes, prev_array_props_);  // NOLINT
    op_node->rank_ = rank_;

    // Add new exotic previous array nodes
    for (auto& tup : new_exotic_prev_array_nodes) {
        op_node->RegisterExoticPreviousArrayNodes(std::move(std::get<0>(tup)), std::move(std::get<1>(tup)));
    }

    // Register backward function entries.
    // Both backward_entries_ and next_array_nodes_ members will be set.
    for (const internal::OpNodeBackwardEntry& backward_entry : backward_entries_) {
        // Collect new next_array_nodes and exotic_next_array_nodes.
        // Old next_array_nodes of "this" graph will be exotic in new op node.
        std::vector<std::shared_ptr<ArrayNode>> new_next_array_nodes;
        std::vector<std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>> new_exotic_next_array_nodes;
        {
            for (const std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>& array_nodes :
                 backward_entry.exotic_next_array_nodes()) {
                const GraphId& local_graph_id = std::get<0>(array_nodes);
                assert(local_graph_id != graph_id_);
                if (local_graph_id == other_graph_id) {
                    new_next_array_nodes = std::get<1>(array_nodes);
                } else {
                    new_exotic_next_array_nodes.emplace_back(array_nodes);
                }
            }
            assert(!new_next_array_nodes.empty());

            // Next array nodes of current "this" graph is new "exotic" next array nodes.
            new_exotic_next_array_nodes.emplace_back(
                    std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>{graph_id_, backward_entry.GetNextArrayNodes()});
        }

        // Register new backward function entry and new next array nodes.
        internal::OpNodeBackwardEntry& new_backward_entry =
                op_node->RegisterBackwardFunction(std::move(new_next_array_nodes), backward_entry.backward_func());

        // Register new exotic next array nodes.
        for (std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>>& array_nodes : new_exotic_next_array_nodes) {
            new_backward_entry.AddExoticNextArrayNode(std::move(array_nodes));
        }
    }
    op_node->AssertConsistency();
    return op_node;
}

}  // namespace xchainer
