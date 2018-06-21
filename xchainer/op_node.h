#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "xchainer/backward.h"
#include "xchainer/graph.h"

namespace xchainer {

class Array;
class ArrayNode;
class BackwardContext;
class Device;

namespace internal {

class OpNodeBackwardEntry {
public:
    OpNodeBackwardEntry(std::vector<size_t> next_array_node_indices, BackwardFunction backward_func);

    size_t next_array_node_count() const { return next_array_node_indices_.size(); }

    gsl::span<const size_t> next_array_node_indices() const { return next_array_node_indices_; }

    const BackwardFunction& backward_func() const { return backward_func_; }

private:
    std::vector<size_t> next_array_node_indices_;
    BackwardFunction backward_func_;
};

}  // namespace internal

class OpNode {
public:
    OpNode() = default;
    explicit OpNode(std::string name, const std::vector<std::shared_ptr<ArrayNode>>& prev_array_nodes);

    OpNode(const OpNode&) = delete;
    OpNode(OpNode&&) = delete;
    OpNode& operator=(const OpNode&) = delete;
    OpNode& operator=(OpNode&&) = delete;

    void RegisterBackwardFunction(
            gsl::span<std::reference_wrapper<std::shared_ptr<ArrayNode>>> next_array_nodes, BackwardFunction backward_func);

    void Unchain() {
        backward_entries_.clear();
        next_array_nodes_.clear();
    }

    std::string name() const { return name_; }

    gsl::span<std::shared_ptr<ArrayNode>> next_array_nodes();

    gsl::span<const std::shared_ptr<ArrayNode>> next_array_nodes() const;

    gsl::span<internal::OpNodeBackwardEntry> backward_entries() { return backward_entries_; }

    gsl::span<const internal::OpNodeBackwardEntry> backward_entries() const { return backward_entries_; }

    size_t next_array_node_count() const { return next_array_nodes_.size(); }

    size_t prev_node_count() const { return prev_array_nodes_.size(); }

    const std::vector<std::weak_ptr<ArrayNode>>& prev_array_nodes() const { return prev_array_nodes_; }

    int64_t rank() const { return rank_; }

    GraphId graph_id() const { return graph_id_; }

private:
    std::string name_;
    GraphId graph_id_;
    int64_t rank_{0};
    std::vector<std::shared_ptr<ArrayNode>> next_array_nodes_;
    std::vector<std::weak_ptr<ArrayNode>> prev_array_nodes_;

    std::vector<internal::OpNodeBackwardEntry> backward_entries_;
};

}  // namespace xchainer
