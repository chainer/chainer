// This header file is needed to include in every compilation unit of the Python extension module.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nonstd/optional.hpp>

// Optional type caster
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<nonstd::optional<T>> : optional_caster<nonstd::optional<T>> {};

}  // namespace detail
}  // namespace pybind11
