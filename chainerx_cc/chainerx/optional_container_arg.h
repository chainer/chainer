#pragma once

#include <initializer_list>
#include <utility>

#include <absl/types/optional.h>

#include "chainerx/macro.h"

namespace chainerx {

template <typename Container>
class OptionalContainerArg {
public:
    using T = typename Container::value_type;

    // ctor for null value
    OptionalContainerArg(absl::nullopt_t /*nullopt*/) : opt_{absl::nullopt} {}  // NOLINT

    // ctor for single value
    OptionalContainerArg(const T& value) : opt_{Container{{value}}} {}  // NOLINT

    // ctors for vector value
    OptionalContainerArg(std::initializer_list<T> list) : opt_{Container{list.begin(), list.end()}} {}
    OptionalContainerArg(const Container& v) : opt_{v} {}  // NOLINT
    OptionalContainerArg(Container&& v) : opt_{std::move(v)} {}  // NOLINT

    bool has_value() const { return opt_.has_value(); }

    Container& value() { return opt_.value(); }

    const Container& value() const { return opt_.value(); }

    Container& operator*() {
        CHAINERX_ASSERT(has_value());
        return *opt_;
    }

    const Container& operator*() const {
        CHAINERX_ASSERT(has_value());
        return *opt_;
    }

    Container* operator->() {
        CHAINERX_ASSERT(has_value());
        return &*opt_;
    }

    const Container* operator->() const {
        CHAINERX_ASSERT(has_value());
        return &*opt_;
    }

    explicit operator bool() const { return has_value(); }

private:
    absl::optional<Container> opt_;
};

}  // namespace chainerx
