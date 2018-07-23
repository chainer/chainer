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

    // Returns the graph ID.
    const GraphId& graph_id() const { return graph_id_; }

    // Returns the array body. It returns nullptr if the array body is no longer alive.
    std::shared_ptr<const internal::ArrayBody> GetBody() const { return body_.lock(); }

    // Returns the array body. It returns nullptr if the array body is no longer alive.
    std::shared_ptr<internal::ArrayBody> GetBody() { return body_.lock(); }

private:
    // body_ is set by this function.
    friend const std::shared_ptr<ArrayNode>& internal::ArrayBody::AddNode(const std::shared_ptr<ArrayBody>&, std::shared_ptr<ArrayNode>);

    std::weak_ptr<internal::ArrayBody> body_;
    std::shared_ptr<OpNode> next_op_node_;
    Shape shape_;
    Dtype dtype_;
    Device& device_;
    GraphId graph_id_;
};

}  // namespace xchainer
