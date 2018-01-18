#include "xchainer/python/array.h"

#include <algorithm>
#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;

Dtype NumpyDtypeToDtype(const py::dtype& npdtype) {
    switch (npdtype.kind()) {
        case 'b':
            return Dtype::kBool;
        case 'i':
            switch (npdtype.itemsize()) {
                case 1:
                    return Dtype::kInt8;
                case 2:
                    return Dtype::kInt16;
                case 4:
                    return Dtype::kInt32;
                case 8:
                    return Dtype::kInt64;
                default:
                    break;
            }
            break;
        case 'u':
            switch (npdtype.itemsize()) {
                case 1:
                    return Dtype::kUInt8;
                default:
                    break;
            }
            break;
        case 'f':
            switch (npdtype.itemsize()) {
                case 4:
                    return Dtype::kFloat32;
                case 8:
                    return Dtype::kFloat64;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    throw DtypeError("unsupported NumPy dtype");
}

Array MakeArray(const Shape& shape, Dtype dtype, py::list list) {
    auto total_size = shape.total_size();
    auto bytes = GetElementSize(dtype) * total_size;
    if (static_cast<size_t>(total_size) != list.size()) {
        throw DimensionError("Invalid data length");
    }

    // Allocate a buffer and copy data
    std::shared_ptr<void> ptr = std::make_unique<uint8_t[]>(bytes);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        std::transform(list.begin(), list.end(), static_cast<T*>(ptr.get()), [](auto& item) { return py::cast<T>(item); });
    });
    return Array::FromBuffer(shape, dtype, ptr);
}

Array MakeArray(py::array array) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw DimensionError("cannot convert non-contiguous NumPy array to Array");
    }

    Dtype dtype = NumpyDtypeToDtype(array.dtype());
    py::buffer_info info = array.request();
    Shape shape(info.shape);

    // data holds the copy of py::array which in turn references the NumPy array and the buffer is therefore not released
    std::shared_ptr<void> data(std::make_shared<py::array>(std::move(array)), array.mutable_data());

    return Array::FromBuffer(shape, dtype, data);
}

py::buffer_info MakeNumpyArrayFromArray(Array& self) {
    if (!self.is_contiguous()) {
        throw DimensionError("cannot convert non-contiguous Array to NumPy array");
    }

    int64_t itemsize{GetElementSize(self.dtype())};
    const Shape& shape = self.shape();

    // compute C-contiguous strides
    size_t ndim = self.ndim();
    std::vector<size_t> strides(ndim);
    if (ndim > 0) {
        std::partial_sum(shape.crbegin(), shape.crend() - 1, strides.rbegin() + 1, std::multiplies<size_t>());
        strides.back() = 1;
        std::transform(strides.crbegin(), strides.crend(), strides.rbegin(), [&itemsize](size_t item) { return item * itemsize; });
    }

    return py::buffer_info(self.data().get(), itemsize, std::string(1, GetCharCode(self.dtype())), ndim, shape, strides);
}

void InitXchainerArray(pybind11::module& m) {
    py::class_<Array>{m, "Array", py::buffer_protocol()}
        .def(py::init(py::overload_cast<const Shape&, Dtype, py::list>(&MakeArray)))
        .def(py::init(py::overload_cast<py::array>(&MakeArray)))
        .def_buffer(&MakeNumpyArrayFromArray)
        .def("view", &Array::MakeView)
        .def(py::self += py::self)
        .def(py::self *= py::self)
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def("__repr__", static_cast<std::string (Array::*)() const>(&Array::ToString))
        .def("copy", &Array::Copy)
        .def_property("requires_grad", &Array::requires_grad, &Array::set_requires_grad)
        .def_property("grad", &Array::grad,
                      [](Array& self, Array* grad) {
                          if (grad) {
                              self.set_grad(grad->MakeView());
                          } else {
                              self.ClearGrad();
                          }
                      })
        .def_property_readonly("dtype", &Array::dtype)
        .def_property_readonly("element_bytes", &Array::element_bytes)
        .def_property_readonly("is_contiguous", &Array::is_contiguous)
        .def_property_readonly("ndim", &Array::ndim)
        .def_property_readonly("offset", &Array::offset)
        .def_property_readonly("shape", &Array::shape)
        .def_property_readonly("total_bytes", &Array::total_bytes)
        .def_property_readonly("total_size", &Array::total_size)
        .def_property_readonly("_debug_data_memory_address",  // These methods starting with `_debug_` are stubs for testing
                               [](const Array& self) { return reinterpret_cast<std::uintptr_t>(self.data().get()); })
        .def_property_readonly("_debug_flat_data", [](const Array& self) {
            py::list list;
            auto size = self.total_size();

            // Copy data into the list
            VisitDtype(self.dtype(), [&](auto pt) {
                using T = typename decltype(pt)::type;
                const T& data = *std::static_pointer_cast<const T>(self.data());
                for (int64_t i = 0; i < size; ++i) {
                    list.append((&data)[i]);
                }
            });

            return list;
        });
}

}  // namespace xchainer
