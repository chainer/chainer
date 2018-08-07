#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include <nonstd/optional.hpp>

#include "xchainer/array_body.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/op_node.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace internal {

class ArrayNode {
public:
    ArrayNode(const Shape& shape, Dtype dtype, Device& device, BackpropId backprop_id)
        : shape_{shape}, dtype_{dtype}, device_{device}, backprop_id_{std::move(backprop_id)} {}

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
        assert(backprop_id() == next_op_node->backprop_id());
        next_op_node_ = std::move(next_op_node);
    }

    int64_t rank() const {
        if (next_op_node_ == nullptr) {
            return 0;
        }
        return next_op_node_->rank();
    }

    // Returns the backprop ID.
    const BackpropId& backprop_id() const { return backprop_id_; }

    const std::weak_ptr<ArrayBody>& weak_body() const { return weak_body_; }

private:
    // weak_body_ is set by this function.
    friend const std::shared_ptr<ArrayNode>& ArrayBody::AddNode(const std::shared_ptr<ArrayBody>&, std::shared_ptr<ArrayNode>);

    std::weak_ptr<ArrayBody> weak_body_;
    std::shared_ptr<OpNode> next_op_node_;
    Shape shape_;
    Dtype dtype_;
    Device& device_;
    BackpropId backprop_id_;
};

}  // namespace internal
}  // namespace xchainer
