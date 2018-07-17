#pragma once

#include <cstdint>
#include <functional>

#include "xchainer/hash_combine.h"

namespace xchainer {

using GraphSubId = uint64_t;

class Context;

class GraphId {
public:
    GraphId(Context& context, GraphSubId sub_id) : context_{context}, sub_id_{sub_id} {}

    bool operator==(const GraphId& other) const { return &context_ == &other.context_ && sub_id_ == other.sub_id_; }

    bool operator!=(const GraphId& other) const { return !operator==(other); }

    Context& context() const { return context_; }
    GraphSubId sub_id() const { return sub_id_; }

private:
    Context& context_;
    GraphSubId sub_id_;
};

// Used to represent any graph (id).
class AnyGraph {};

}  // namespace xchainer

namespace std {

template <>
struct hash<xchainer::GraphId> {
    size_t operator()(const xchainer::GraphId& graph_id) const {
        size_t seed = std::hash<xchainer::Context*>()(&graph_id.context());
        xchainer::internal::HashCombine(seed, std::hash<xchainer::GraphSubId>()(graph_id.sub_id()));
        return seed;
    }
};

}  // namespace std
