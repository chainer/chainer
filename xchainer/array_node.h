#pragma once

#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"

namespace xchainer {

class OpNode;

class ArrayNode {
public:
    ArrayNode() = default;

    const std::shared_ptr<OpNode>& next_node() { return next_node_; }
    std::shared_ptr<const OpNode> next_node() const { return next_node_; }

    void set_next_node(std::shared_ptr<OpNode> next_node) { next_node_ = std::move(next_node); }

    const nonstd::optional<Array>& grad() const { return grad_; }

    void set_grad(const Array& grad) { grad_.emplace(grad); };
    void set_grad(Array&& grad) { grad_.emplace(grad); };

    void ClearGrad() { grad_ = nonstd::nullopt; }

private:
    std::shared_ptr<OpNode> next_node_;
    nonstd::optional<Array> grad_;
};

}  // xchainer
