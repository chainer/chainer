// This header file is needed to include in every compilation unit of the Python extension module.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

#pragma once

#include <vector>

#include <nonstd/optional.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xchainer/axes.h"
#include "xchainer/ndim_vector.h"

// Optional type caster
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<nonstd::optional<T>> : optional_caster<nonstd::optional<T>> {};

}  // namespace detail
}  // namespace pybind11

namespace xchainer {
namespace python {
namespace internal {

inline OptionalAxes ToAxes(const nonstd::optional<std::vector<int8_t>>& vec) {
    if (vec.has_value()) {
        return Axes{vec->begin(), vec->end()};
    } else {
        return nonstd::nullopt;
    }
}

inline OptionalAxes ToAxes(const nonstd::optional<int8_t>& vec) {
    if (vec.has_value()) {
        return Axes{*vec};
    } else {
        return nonstd::nullopt;
    }
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
