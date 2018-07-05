#pragma once

#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <utility>

#include "xchainer/hash_combine.h"

namespace xchainer {

class GraphId {
public:
    enum Type { kNamed, kDefault, kAny };

    GraphId() = default;
    GraphId(Type type) : type_{type} {}
    GraphId(const char* graph_id) : type_{kNamed}, graph_id_{graph_id} {}
    GraphId(std::string graph_id) : type_{kNamed}, graph_id_{std::move(graph_id)} {}

    bool operator==(const GraphId& other) const;

    Type type() const { return type_; }

    const std::string& graph_id() const { return graph_id_; }

    std::string& graph_id() { return graph_id_; }

    std::string ToString() const;

private:
    Type type_{};
    std::string graph_id_{};
};

std::ostream& operator<<(std::ostream& os, const GraphId& graph_id);

}  // namespace xchainer

namespace std {

template <>
struct hash<::xchainer::GraphId> {
    size_t operator()(const ::xchainer::GraphId& graph_id) const {
        size_t seed = hash<std::string>()(graph_id.graph_id());
        ::xchainer::internal::HashCombine(seed, hash<int>()(static_cast<int>(graph_id.type())));
        return seed;
    }
};

}  // namespace std
