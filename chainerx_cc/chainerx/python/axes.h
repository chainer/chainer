#pragma once

#include <cstdint>
#include <vector>

#include <absl/types/optional.h>
#include <pybind11/pybind11.h>

#include "chainerx/axes.h"

namespace chainerx {
namespace python {
namespace python_internal {

Axes ToAxes(const pybind11::tuple& tup);

inline OptionalAxes ToAxes(const absl::optional<std::vector<int8_t>>& vec) {
    if (vec.has_value()) {
        return Axes{vec->begin(), vec->end()};
    }
    return absl::nullopt;
}

inline OptionalAxes ToAxes(absl::optional<int8_t> vec) {
    if (vec.has_value()) {
        return Axes{*vec};
    }
    return absl::nullopt;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
