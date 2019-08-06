// This header file is needed to include in every compilation unit of the Python extension module.
// http://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

#pragma once

#include <absl/types/optional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Optional type caster
// http://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<absl::optional<T>> : optional_caster<absl::optional<T>> {};

}  // namespace detail
}  // namespace pybind11
