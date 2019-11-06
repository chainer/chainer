#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "chainerx/array_body.h"

namespace chainerx {
namespace python {
namespace python_internal {

// TODO(beam2d): The current binding has an overhead on wrapping ArrayBodyPtr by Array, which copies shared_ptr. One
// simple way to avoid this overhead is to use reinterpret_cast<Array&>(ptr). This cast is valid if ArrayBodyPtr (i.e.,
// shared_ptr) satisfies "standard layout" conditions. We can test if ArrayBodyPtr satisfies these conditions by
// std::is_standard_layout (see http://en.cppreference.com/w/cpp/types/is_standard_layout#Notes).

using ArrayBody = internal::ArrayBody;
using ArrayBodyPtr = std::shared_ptr<ArrayBody>;
using ConstArrayBodyPtr = std::shared_ptr<const ArrayBody>;

ArrayBodyPtr MakeArray(pybind11::handle object, pybind11::handle dtype, bool copy, pybind11::handle device);

ArrayBodyPtr MakeArray(pybind11::handle object, absl::optional<Dtype> dtype, bool copy, Device& device);

pybind11::tuple ToTuple(const std::vector<Array>& ary);

std::vector<ArrayBodyPtr> ToArrayBodyPtr(const std::vector<Array>& ary);

// Makes an array from a NumPy array. Shape, dtype, strides will be kept.
ArrayBodyPtr MakeArrayFromNumpyArray(pybind11::array array, Device& device);

void InitChainerxArray(pybind11::module& m);

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
