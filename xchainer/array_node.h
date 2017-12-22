#pragma once

#include <memory>

namespace xchainer {

class OpNode;

class ArrayNode {
public:
    ArrayNode() = default;

    const std::shared_ptr<OpNode>& next_node() { return next_node_; }
    std::shared_ptr<const OpNode> next_node() const { return next_node_; }

    void set_next_node(std::shared_ptr<OpNode> next_node) { next_node_ = std::move(next_node); }


private:
    std::shared_ptr<OpNode> next_node_;
};

}  // xchainer
