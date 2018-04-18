// This header file is needed to include in every compilation unit of the Python extension module.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

#pragma once

#include <vector>

#include <nonstd/optional.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

template <typename T>
nonstd::optional<NdimVector<T>> ToNdimVector(const nonstd::optional<std::vector<T>>& vec) {
    if (vec.has_value()) {
        return NdimVector<T>{vec->begin(), vec->end()};
    } else {
        return nonstd::nullopt;
    }
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
