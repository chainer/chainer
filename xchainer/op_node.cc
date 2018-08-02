#include "xchainer/op_node.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

ArrayProps::ArrayProps(const Array& array) : shape{array.shape()}, dtype{array.dtype()}, device{array.device()} {}
ArrayProps::ArrayProps(const ArrayNode& array_node) : shape{array_node.shape()}, dtype{array_node.dtype()}, device{array_node.device()} {}

OpNodeBackwardEntry::OpNodeBackwardEntry(OpNode& op_node, std::vector<size_t> next_array_node_indices, BackwardFunction backward_func)
    : op_node_{op_node}, next_array_node_indices_{std::move(next_array_node_indices)}, backward_func_{std::move(backward_func)} {}

void OpNodeBackwardEntry::AddExoticNextArrayNode(std::tuple<GraphId, std::vector<std::shared_ptr<ArrayNode>>> next_array_nodes) {
    assert(std::get<0>(next_array_nodes) != op_node_.graph_id());
    exotic_next_array_nodes_.emplace_back(std::move(next_array_nodes));
}

std::shared_ptr<ArrayNode> FabricatePrevArrayNode(std::shared_ptr<OpNode> op_node, size_t prev_array_node_index) {
    assert(prev_array_node_index < op_node->prev_array_node_count());
    assert(op_node->prev_array_nodes()[prev_array_node_index].expired());

    const ArrayProps& props = op_node->GetPrevArrayProps(prev_array_node_index);
    auto prev_array_node = std::make_shared<ArrayNode>(props.shape, props.dtype, props.device, op_node->graph_id());

    op_node->prev_array_nodes()[prev_array_node_index] = prev_array_node;
    prev_array_node->set_next_op_node(std::move(op_node));

    return prev_array_node;
}

// static
std::shared_ptr<OpNode> OpNode::CreateWithPrevArrayNodes(
        std::string name, GraphId graph_id, size_t input_count, const std::vector<ConstArrayRef>& outputs) {
    // Trick to use make_shared with private ctor
    struct OpNodeWithPublicCtor : OpNode {
        OpNodeWithPublicCtor(std::string name, GraphId graph_id, size_t input_count) : OpNode{std::move(name), graph_id, input_count} {}
    };
    std::shared_ptr<OpNode> op_node = std::make_shared<OpNodeWithPublicCtor>(std::move(name), graph_id, input_count);

    for (const Array& out : outputs) {
        const std::shared_ptr<ArrayBody>& out_body = GetArrayBody(out);
        assert(!out_body->HasArrayNode(graph_id));
        const std::shared_ptr<ArrayNode>& prev_array_node = ArrayBody::CreateArrayNode(out_body, graph_id);
        op_node->prev_array_props_.emplace_back(*prev_array_node);
        op_node->prev_array_nodes_.emplace_back(prev_array_node);
        prev_array_node->set_next_op_node(op_node);
    }
    op_node->AssertConsistency();
    return op_node;
}

OpNode::OpNode(std::string name, GraphId graph_id, size_t next_array_node_count)
    : name_{std::move(name)},
      graph_id_{graph_id},
      next_array_nodes_{next_array_node_count}  // Initialize with nullptrs
{}

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
    for (size_t i_prev = 0; i_prev < prev_array_node_count(); ++i_prev) {
        nonstd::optional<ArrayBody*> prev_array_body{};
        for (const auto& tup : outer_graphs_prev_array_nodes_) {
            const std::vector<std::shared_ptr<ArrayNode>>& vec = std::get<1>(tup);
            const std::shared_ptr<ArrayNode>& prev_array_node = vec[i_prev];
            std::shared_ptr<ArrayBody> body = prev_array_node->weak_body().lock();
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
        return arr_node == nullptr || arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

const std::vector<std::shared_ptr<ArrayNode>>& OpNode::next_array_nodes() const {
    assert(std::all_of(next_array_nodes_.begin(), next_array_nodes_.end(), [this](const std::shared_ptr<ArrayNode>& arr_node) {
        return arr_node == nullptr || arr_node->graph_id() == graph_id_;
    }));
    return next_array_nodes_;
}

OpNodeBackwardEntry& OpNode::RegisterBackwardFunction(
        std::vector<std::tuple<size_t, std::shared_ptr<ArrayNode>>> next_array_nodes, BackwardFunction backward_func) {
    AssertConsistency();
    assert(!next_array_nodes.empty());
    assert(std::all_of(next_array_nodes.begin(), next_array_nodes.end(), [this](const auto& tup) {
        const std::shared_ptr<ArrayNode>& next_array_node = std::get<1>(tup);
        // next_array_node could be nullptr, if the corresponding input array does not require grad.
        return next_array_node == nullptr || next_array_node->graph_id() == graph_id_;
    }));

    // Update the rank of op node
    for (const auto& tup : next_array_nodes) {
        const std::shared_ptr<ArrayNode>& next_array_node = std::get<1>(tup);
        if (next_array_node != nullptr) {
            rank_ = std::max(rank_, next_array_node->rank() + 1);
        }
    }

    // Store next nodes and record indices of them
    std::vector<size_t> next_array_node_indices;
    next_array_node_indices.reserve(next_array_nodes.size());
    for (auto& tup : next_array_nodes) {
        size_t next_index = std::get<0>(tup);
        std::shared_ptr<ArrayNode>& next_array_node = std::get<1>(tup);

        next_array_node_indices.emplace_back(next_index);
        if (next_array_node != nullptr) {
            assert(gsl::at(next_array_nodes_, next_index) == nullptr);
            gsl::at(next_array_nodes_, next_index) = std::move(next_array_node);
        }
    }

    // Add backward entry
    backward_entries_.emplace_back(*this, std::move(next_array_node_indices), std::move(backward_func));

    AssertConsistency();
    return backward_entries_.back();
}

void OpNode::AddEdgesToPreviousArrayNodesOfOuterGraph(
        const GraphId& outer_graph_id, std::vector<std::shared_ptr<ArrayNode>> outer_graphs_prev_array_nodes) {
    AssertConsistency();
    assert(outer_graph_id < graph_id_);
    assert(outer_graphs_prev_array_nodes.size() == prev_array_props_.size());
    assert(std::all_of(
            outer_graphs_prev_array_nodes.begin(),
            outer_graphs_prev_array_nodes.end(),
            [&outer_graph_id](const std::shared_ptr<ArrayNode>& prev) { return prev->graph_id() == outer_graph_id; }));

    outer_graphs_prev_array_nodes_.emplace_back(outer_graph_id, std::move(outer_graphs_prev_array_nodes));

    AssertConsistency();
}

}  // namespace internal
}  // namespace xchainer
