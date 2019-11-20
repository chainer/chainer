#pragma once

#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>

#include "chainerx/constant.h"
#include "chainerx/dims.h"
#include "chainerx/error.h"

namespace chainerx {
namespace python {
namespace python_internal {

// Returns a StackVector of length size filled with the given item.
template <typename T>
StackVector<T, kMaxNdim> ToStackVector(pybind11::int_ item, size_t size) {
    StackVector<T, kMaxNdim> out;
    std::fill_n(std::back_inserter(out), size, py::cast<T>(item));
    return out;
}

template <typename T>
StackVector<T, kMaxNdim> ToStackVector(const pybind11::tuple& tup) {
    StackVector<T, kMaxNdim> out;
    std::transform(tup.begin(), tup.end(), std::back_inserter(out), [](auto& item) { return py::cast<T>(item); });
    return out;
}

// If the given handle is a scalar, returns a StackVector of length size filled with that scalar.
// Else if the handle is a tuple, the tuple is converted to a StackVector and the size argument is ignored.
// It is used to e.g. pre-process convolution arguments such as stride and padding.
template <typename T>
StackVector<T, kMaxNdim> ToStackVector(pybind11::handle handle, size_t size) {
    if (py::isinstance<py::int_>(handle)) {
        return ToStackVector<T>(py::cast<py::int_>(handle), size);
    }
    if (py::isinstance<py::tuple>(handle)) {
        return ToStackVector<T>(py::cast<py::tuple>(handle));
    }
    // TODO(hvy): Extend with additional types as necessary.
    throw py::type_error{"Only int and tuple is allowed."};
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
