#pragma once

#include <initializer_list>
#include <utility>

#include <nonstd/optional.hpp>

namespace xchainer {
namespace optional_vector_detail {

enum class OptionalState {
    kNull = 0,
    kSingle,
    kVector,
};
}

template <typename Vector>
class OptionalVector {
public:
    using T = typename Vector::value_type;
    using OptionalState = optional_vector_detail::OptionalState;

    // ctor for null value
    OptionalVector(nonstd::nullopt_t) : state_{OptionalState::kNull} {}  // NOLINT(runtime/explicit)

    // ctor for single value
    OptionalVector(const T& value) : v_{value}, state_{OptionalState::kSingle} {}  // NOLINT(runtime/explicit)

    // ctors for vector value
    OptionalVector(std::initializer_list<T> list)
        : v_{list.begin(), list.end()}, state_{OptionalState::kVector} {}             // NOLINT(runtime/explicit)
    OptionalVector(const Vector& v) : v_{v}, state_{OptionalState::kVector} {}        // NOLINT(runtime/explicit)
    OptionalVector(Vector&& v) : v_{std::move(v)}, state_{OptionalState::kVector} {}  // NOLINT(runtime/explicit)

    bool has_value() const { return state_ != OptionalState::kNull; }

    bool is_single() const { return state_ == OptionalState::kSingle; }

    bool is_vector() const { return state_ == OptionalState::kVector; }

    const Vector& vector() const {
        assert(is_vector());
        return v_;
    }

    const Vector& as_vector() const {
        assert(has_value());
        return v_;
    }

    const T& single() const {
        assert(is_single());
        assert(!v_.empty());
        return v_[0];
    }

private:
    Vector v_;
    optional_vector_detail::OptionalState state_;
};

}  // namespace xchainer
