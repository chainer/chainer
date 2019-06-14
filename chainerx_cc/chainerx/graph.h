#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

#include "chainerx/error.h"
#include "chainerx/hash_combine.h"

namespace chainerx {

using BackpropOrdinal = uint64_t;

class Context;

class BackpropId {
public:
    ~BackpropId() = default;

    BackpropId(const BackpropId&) = default;
    BackpropId(BackpropId&&) = default;
    BackpropId& operator=(const BackpropId&) = default;
    BackpropId& operator=(BackpropId&&) = default;

    bool operator==(const BackpropId& other) const { return &context_.get() == &other.context_.get() && ordinal_ == other.ordinal_; }

    bool operator!=(const BackpropId& other) const { return !operator==(other); }

    bool operator<(const BackpropId& other) const { return CompareImpl<std::less<BackpropOrdinal>>(other); }

    bool operator<=(const BackpropId& other) const { return CompareImpl<std::less_equal<BackpropOrdinal>>(other); }

    bool operator>(const BackpropId& other) const { return CompareImpl<std::greater<BackpropOrdinal>>(other); }

    bool operator>=(const BackpropId& other) const { return CompareImpl<std::greater_equal<BackpropOrdinal>>(other); }

    Context& context() const { return context_; }

    BackpropOrdinal ordinal() const { return ordinal_; }

    // Returns the backprop name.
    // ChainerxError is thrown if the backprop ID is expired or non-existent in the associated context.
    std::string GetName() const;

    // Throws ChainerxError if this backprop ID has already been released.
    void CheckValid() const;

private:
    // A BackpropId is always constructed by a Context.
    friend class Context;

    BackpropId(Context& context, BackpropOrdinal ordinal) : context_{context}, ordinal_{ordinal} {}

    template <typename Compare>
    bool CompareImpl(const BackpropId& other) const {
        if (&context_.get() != &other.context_.get()) {
            throw ContextError{"Cannot compare backprop ids with different contexts."};
        }
        return Compare{}(ordinal_, other.ordinal_);
    }

    // Using reference_wrapper to make this class move assignable
    std::reference_wrapper<Context> context_;

    BackpropOrdinal ordinal_;
};

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id);

// Used to represent any graph (id).
class AnyGraph {};

}  // namespace chainerx

namespace std {

template <>
struct hash<chainerx::BackpropId> {
    size_t operator()(const chainerx::BackpropId& backprop_id) const {
        size_t seed = std::hash<chainerx::Context*>()(&backprop_id.context());
        chainerx::internal::HashCombine(seed, std::hash<chainerx::BackpropOrdinal>()(backprop_id.ordinal()));
        return seed;
    }
};

}  // namespace std
