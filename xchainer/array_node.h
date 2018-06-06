#pragma once

#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/op_node.h"

namespace xchainer {

class ArrayNode {
public:
    ArrayNode(const Shape& shape, Dtype dtype, Device& device, GraphId graph_id)
        : shape_{shape}, dtype_{dtype}, device_{device}, graph_id_{std::move(graph_id)} {}

    const Shape& shape() const { return shape_; }

    Dtype dtype() const { return dtype_; }

    Device& device() const { return device_; }

    const std::shared_ptr<OpNode>& next_node() { return next_node_; }
    std::shared_ptr<const OpNode> next_node() const { return next_node_; }
    std::shared_ptr<OpNode> move_next_node() { return std::move(next_node_); }

    void set_next_node(std::shared_ptr<OpNode> next_node) {
        assert(next_node != nullptr);
        assert(next_node_ == nullptr);
        assert(graph_id() == next_node->graph_id());
        next_node_ = std::move(next_node);
        rank_ = next_node_->rank() + 1;
    }

    int64_t rank() const { return rank_; }

    const nonstd::optional<Array>& grad() const noexcept { return grad_; }

    void set_grad(const Array& grad) { grad_ = grad; }

    void set_grad(Array&& grad) { grad_ = std::move(grad); }

    void accumulate_grad(Array&& grad) {
        if (grad_.has_value()) {
            grad_ = *grad_ + grad;
        } else {
            grad_ = std::move(grad);
        }
    }

    GraphId graph_id() const { return graph_id_; }

    void set_graph_id(GraphId graph_id) { graph_id_ = std::move(graph_id); }

    void ClearGrad() noexcept { grad_.reset(); }

private:
    std::shared_ptr<OpNode> next_node_;
    int64_t rank_{0};
    Shape shape_;
    Dtype dtype_;
    Device& device_;
    nonstd::optional<Array> grad_;
    GraphId graph_id_;
};

}  // namespace xchainer
