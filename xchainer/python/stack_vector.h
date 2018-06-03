#pragma once

#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>

#include "xchainer/error.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace python {
namespace internal {

// Returns a StackVector of length size filled with the given item.
template <typename T, int8_t Ndim>
StackVector<T, Ndim> ToStackVector(pybind11::int_ item, size_t size) {
    StackVector<T, Ndim> out;
    std::fill_n(std::back_inserter(out), size, py::cast<T>(item));
    return out;
}

template <typename T, int8_t Ndim>
StackVector<T, Ndim> ToStackVector(const pybind11::tuple& tup) {
    StackVector<T, Ndim> out;
    std::transform(tup.begin(), tup.end(), std::back_inserter(out), [](auto& item) { return py::cast<T>(item); });
    return out;
}

// If the given handle is a scalar, returns a StackVector of length size filled with that scalar.
// Else if the handle is a tuple, the tuple is converted to a StackVector and the size argument is ignored.
// It is used to e.g. pre-process convolution arguments such as stride and padding.
template <typename T, int8_t Ndim>
StackVector<T, Ndim> ToStackVector(pybind11::handle handle, size_t size) {
    if (py::isinstance<py::int_>(handle)) {
        return ToStackVector<T, Ndim>(py::cast<py::int_>(handle), size);
    } else if (py::isinstance<py::tuple>(handle)) {
        return ToStackVector<T, Ndim>(py::cast<py::tuple>(handle));
    }
    throw py::type_error{"Only py::int_ and py::tuple can be converted into a StackVector."};
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
