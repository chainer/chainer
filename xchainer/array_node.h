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

    int64_t rank() const { return rank_; }

    void set_rank(int64_t rank) { rank_ = rank; }

    const nonstd::optional<Array>& grad() const noexcept { return grad_; }

    void set_grad(Array grad) { grad_.emplace(std::move(grad)); };

    void ClearGrad() noexcept { grad_.reset(); }

private:
    std::shared_ptr<OpNode> next_node_;
    int64_t rank_;
    nonstd::optional<Array> grad_;
};

}  // xchainer
