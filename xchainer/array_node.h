#pragma once

#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/op_node.h"
#include "xchainer/shape.h"

namespace xchainer {

class ArrayNode {
public:
    ArrayNode(const Shape& shape, Dtype dtype, Device& device, GraphId graph_id)
        : shape_{shape}, dtype_{dtype}, device_{device}, graph_id_{std::move(graph_id)} {}

    ArrayNode(const ArrayNode&) = delete;
    ArrayNode(ArrayNode&&) = delete;
    ArrayNode& operator=(const ArrayNode&) = delete;
    ArrayNode& operator=(ArrayNode&&) = delete;

    const Shape& shape() const { return shape_; }

    Dtype dtype() const { return dtype_; }

    Device& device() const { return device_; }

    const std::shared_ptr<OpNode>& next_op_node() { return next_op_node_; }
    std::shared_ptr<const OpNode> next_op_node() const { return next_op_node_; }
    std::shared_ptr<OpNode> move_next_op_node() { return std::move(next_op_node_); }

    void set_next_op_node(std::shared_ptr<OpNode> next_op_node) {
        assert(next_op_node != nullptr);
        assert(next_op_node_ == nullptr);
        assert(graph_id() == next_op_node->graph_id());
        next_op_node_ = std::move(next_op_node);
    }

    int64_t rank() const {
        if (next_op_node_ == nullptr) {
            return 0;
        }
        return next_op_node_->rank();
    }

    const nonstd::optional<Array>& grad() const noexcept { return grad_; }

    void set_grad(Array grad) {
        CheckGradCompatible(grad);
        grad_ = std::move(grad);
    }

    void AccumulateGrad(Array grad) {
        CheckGradCompatible(grad);
        if (grad_.has_value()) {
            grad_ = *grad_ + grad;
        } else {
            grad_ = std::move(grad);
        }
    }

    GraphId graph_id() const { return graph_id_; }

    void ClearGrad() noexcept { grad_.reset(); }

private:
    void CheckGradCompatible(const Array& grad) {
        CheckEqual(dtype_, grad.dtype());
        CheckEqual(shape_, grad.shape());
        device_.CheckDevicesCompatible(grad);
    }

    std::shared_ptr<OpNode> next_op_node_;
    Shape shape_;
    Dtype dtype_;
    Device& device_;
    nonstd::optional<Array> grad_;
    GraphId graph_id_;
};

}  // namespace xchainer
