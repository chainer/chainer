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
    ArrayNode(const Shape& shape, Dtype dtype, Device& device, GraphId graph_id)
        : shape_{shape}, dtype_{dtype}, device_{device}, graph_id_{std::move(graph_id)} {}

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
        assert(creator_op_node != nullptr);
        assert(creator_op_node_ == nullptr);
        assert(graph_id() == creator_op_node->graph_id());
        creator_op_node_ = std::move(creator_op_node);
    }

    int64_t rank() const {
        if (creator_op_node_ == nullptr) {
            return 0;
        }
        return creator_op_node_->rank();
    }

    // Returns the graph ID.
    const GraphId& graph_id() const { return graph_id_; }

    const std::weak_ptr<ArrayBody>& weak_body() const { return weak_body_; }

private:
    // weak_body_ is set by this function.
    friend const std::shared_ptr<ArrayNode>& ArrayBody::AddNode(const std::shared_ptr<ArrayBody>&, std::shared_ptr<ArrayNode>);

    std::weak_ptr<ArrayBody> weak_body_;
    std::shared_ptr<OpNode> creator_op_node_;
    Shape shape_;
    Dtype dtype_;
    Device& device_;
    GraphId graph_id_;
};

}  // namespace internal
}  // namespace xchainer
