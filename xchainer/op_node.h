#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array_node.h"

namespace xchainer {

class OpNode {
public:
    OpNode() = default;
    OpNode(std::string name, std::vector<std::shared_ptr<const ArrayNode>> next_nodes)
        : name_(std::move(name)), next_nodes_(std::move(next_nodes)) {}

    gsl::span<const std::shared_ptr<const ArrayNode>> next_nodes() const { return gsl::make_span(next_nodes_); }

    std::string name() const { return name_; }

private:
    std::string name_;
    std::vector<std::shared_ptr<const ArrayNode>> next_nodes_;
};

}  // namespace xchainer
