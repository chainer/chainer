#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <absl/types/optional.h>

#include "chainerx/array_body.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/op_node.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace internal {

class ArrayNode {
public:
    ArrayNode(Shape shape, Dtype dtype, Device& device, BackpropId backprop_id)
        : shape_{std::move(shape)}, dtype_{dtype}, device_{device}, backprop_id_{backprop_id} {}

    ~ArrayNode() = default;

    ArrayNode(const ArrayNode&) = delete;
    ArrayNode(ArrayNode&&) = delete;
    ArrayNode& operator=(const ArrayNode&) = delete;
    ArrayNode& operator=(ArrayNode&&) = delete;

    const Shape& shape() const { return shape_; }

    Dtype dtype() const { return dtype_; }

    Device& device() const { return device_; }

    const std::shared_ptr<OpNode>& creator_op_node() { return creator_op_node_; }
    std::shared_ptr<const OpNode> creator_op_node() const { return creator_op_node_; }
    std::shared_ptr<OpNode> move_creator_op_node() { return std::move(creator_op_node_); }

    void set_creator_op_node(std::shared_ptr<OpNode> creator_op_node) {
        CHAINERX_ASSERT(creator_op_node != nullptr);
        CHAINERX_ASSERT(creator_op_node_ == nullptr);
        CHAINERX_ASSERT(backprop_id() == creator_op_node->backprop_id());
        creator_op_node_ = std::move(creator_op_node);
    }

    int64_t rank() const {
        if (creator_op_node_ == nullptr) {
            return 0;
        }
        return creator_op_node_->rank();
    }

    // Returns the backprop ID.
    const BackpropId& backprop_id() const { return backprop_id_; }

    const std::weak_ptr<ArrayBody>& weak_body() const { return weak_body_; }

private:
    // weak_body_ is set by this function.
    friend const std::shared_ptr<ArrayNode>& ArrayBody::AddNode(
            const std::shared_ptr<ArrayBody>& body, std::shared_ptr<ArrayNode> array_node);

    std::weak_ptr<ArrayBody> weak_body_;
    std::shared_ptr<OpNode> creator_op_node_;
    Shape shape_;
    Dtype dtype_;
    Device& device_;
    BackpropId backprop_id_;
};

}  // namespace internal
}  // namespace chainerx
