#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>

#include "xchainer/error.h"
#include "xchainer/hash_combine.h"

namespace xchainer {

using GraphSubId = uint64_t;

class Context;

class BackpropId {
public:
    BackpropId(const BackpropId&) = default;
    BackpropId(BackpropId&&) = default;
    BackpropId& operator=(const BackpropId&) = default;
    BackpropId& operator=(BackpropId&&) = default;

    bool operator==(const BackpropId& other) const { return &context_.get() == &other.context_.get() && sub_id_ == other.sub_id_; }

    bool operator!=(const BackpropId& other) const { return !operator==(other); }

    bool operator<(const BackpropId& other) const { return CompareImpl<std::less<GraphSubId>>(other); }

    bool operator<=(const BackpropId& other) const { return CompareImpl<std::less_equal<GraphSubId>>(other); }

    bool operator>(const BackpropId& other) const { return CompareImpl<std::greater<GraphSubId>>(other); }

    bool operator>=(const BackpropId& other) const { return CompareImpl<std::greater_equal<GraphSubId>>(other); }

    Context& context() const { return context_; }

    GraphSubId sub_id() const { return sub_id_; }

private:
    // A BackpropId is always constructed by a Context.
    friend class Context;

    BackpropId(Context& context, GraphSubId sub_id) : context_{context}, sub_id_{sub_id} {}

    template <typename Compare>
    bool CompareImpl(const BackpropId& other) const {
        if (&context_.get() != &other.context_.get()) {
            throw ContextError{"Cannot compare backprop ids with different contexts."};
        }
        return Compare{}(sub_id_, other.sub_id_);
    }

    // Using reference_wrapper to make this class move assignable
    std::reference_wrapper<Context> context_;

    GraphSubId sub_id_;
};

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id);

// Used to represent any graph (id).
class AnyGraph {};

}  // namespace xchainer

namespace std {

template <>
struct hash<xchainer::BackpropId> {
    size_t operator()(const xchainer::BackpropId& backprop_id) const {
        size_t seed = std::hash<xchainer::Context*>()(&backprop_id.context());
        xchainer::internal::HashCombine(seed, std::hash<xchainer::GraphSubId>()(backprop_id.sub_id()));
        return seed;
    }
};

}  // namespace std
