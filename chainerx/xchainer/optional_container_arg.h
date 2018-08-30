#pragma once

#include <initializer_list>
#include <utility>

#include <nonstd/optional.hpp>

#include "xchainer/macro.h"

namespace xchainer {

template <typename Container>
class OptionalContainerArg {
public:
    using T = typename Container::value_type;

    // ctor for null value
    OptionalContainerArg(nonstd::nullopt_t) : opt_{nonstd::nullopt} {}  // NOLINT(runtime/explicit)

    // ctor for single value
    OptionalContainerArg(const T& value) : opt_{Container{{value}}} {}  // NOLINT(runtime/explicit)

    // ctors for vector value
    OptionalContainerArg(std::initializer_list<T> list) : opt_{Container{list.begin(), list.end()}} {}
    OptionalContainerArg(const Container& v) : opt_{v} {}  // NOLINT(runtime/explicit)
    OptionalContainerArg(Container&& v) : opt_{std::move(v)} {}  // NOLINT(runtime/explicit)

    bool has_value() const { return opt_.has_value(); }

    Container& value() { return opt_.value(); }

    const Container& value() const { return opt_.value(); }

    Container& operator*() {
        XCHAINER_ASSERT(has_value());
        return *opt_;
    }

    const Container& operator*() const {
        XCHAINER_ASSERT(has_value());
        return *opt_;
    }

    Container* operator->() {
        XCHAINER_ASSERT(has_value());
        return &*opt_;
    }

    const Container* operator->() const {
        XCHAINER_ASSERT(has_value());
        return &*opt_;
    }

    explicit operator bool() const { return has_value(); }

private:
    nonstd::optional<Container> opt_;
};

}  // namespace xchainer
