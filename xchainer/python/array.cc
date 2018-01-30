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

namespace {

// TODO(beam2d): The current binding has an overhead on wrapping ArrayBodyPtr by Array, which copies shared_ptr. One
// simple way to avoid this overhead is to use reinterpret_cast<Array&>(ptr). This cast is valid if ArrayBodyPtr (i.e.,
// shared_ptr) satisfies "standard layout" conditions. We can test if ArrayBodyPtr satisfies these conditions by
// std::is_standard_layout (see http://en.cppreference.com/w/cpp/types/is_standard_layout#Notes).

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;
using ConstArrayBodyPtr = std::shared_ptr<const internal::ArrayBody>;

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

ArrayBodyPtr MakeArray(const Shape& shape, Dtype dtype, py::list list) {
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
    return Array::FromBuffer(shape, dtype, ptr).move_body();
}

ArrayBodyPtr MakeArray(py::array array) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw DimensionError("cannot convert non-contiguous NumPy array to Array");
    }

    Dtype dtype = NumpyDtypeToDtype(array.dtype());
    py::buffer_info info = array.request();
    Shape shape(info.shape);

    // data holds the copy of py::array which in turn references the NumPy array and the buffer is therefore not released
    std::shared_ptr<void> data(std::make_shared<py::array>(std::move(array)), array.mutable_data());

    return Array::FromBuffer(shape, dtype, data).move_body();
}

py::buffer_info MakeNumpyArrayFromArray(internal::ArrayBody& self) {
    // Used as a temporary accessor
    Array array{std::move(ArrayBodyPtr(&self, [](internal::ArrayBody* ptr) {
        (void)ptr;  // unused
    }))};

    if (!array.is_contiguous()) {
        throw DimensionError("cannot convert non-contiguous Array to NumPy array");
    }

    int64_t itemsize{GetElementSize(array.dtype())};
    const Shape& shape = array.shape();

    // compute C-contiguous strides
    size_t ndim = array.ndim();
    std::vector<size_t> strides(ndim);
    if (ndim > 0) {
        std::partial_sum(shape.crbegin(), shape.crend() - 1, strides.rbegin() + 1, std::multiplies<size_t>());
        strides.back() = 1;
        std::transform(strides.crbegin(), strides.crend(), strides.rbegin(), [&itemsize](size_t item) { return item * itemsize; });
    }

    return py::buffer_info(array.data().get(), itemsize, std::string(1, GetCharCode(array.dtype())), ndim, shape, strides);
}

}  // namespace

void InitXchainerArray(pybind11::module& m) {
    py::class_<internal::ArrayBody, ArrayBodyPtr>{m, "Array", py::buffer_protocol()}
        .def(py::init(py::overload_cast<const Shape&, Dtype, py::list>(&MakeArray)))
        .def(py::init(py::overload_cast<py::array>(&MakeArray)))
        .def_buffer(&MakeNumpyArrayFromArray)
        .def("view",
             [](const ArrayBodyPtr& self) {
                 // Duplicate the array body
                 return std::make_shared<internal::ArrayBody>(*self);
             })
        .def("__iadd__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} += Array{rhs}).move_body(); })
        .def("__imul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} *= Array{rhs}).move_body(); })
        .def("__add__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} + Array{rhs}).move_body(); })
        .def("__mul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} * Array{rhs}).move_body(); })
        .def("__repr__", [](const ArrayBodyPtr& self) { return Array{self}.ToString(); })
        .def("copy", [](const ArrayBodyPtr& self) { return Array{self}.Copy().move_body(); })
        .def_property("requires_grad", [](const ArrayBodyPtr& self) { return Array{self}.IsGradRequired(); },
                      [](const ArrayBodyPtr& self, bool value) {
                          // TODO(hvy): requires_grad should not be a boolean property but a method that takes a graph id argument, aligning
                          // to the c++ interface. Currently, this property is broken in the sense that once the required_grad flag is set
                          // to true (and an ArrayNode is created internally) it cannot be unset.
                          if (value && !self->HasNode()) {
                              Array{self}.RequireGrad();
                          }
                      })
        .def_property("grad",
                      [](const ArrayBodyPtr& self) -> ConstArrayBodyPtr {
                          if (self->HasNode()) {
                              return Array{self}.GetGrad()->body();
                          } else {
                              return nullptr;
                          }
                      },
                      [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad) {
                          if (grad) {
                              if (self->HasNode()) {
                                  Array{self}.SetGrad(Array{grad});
                              } else {
                                  Array{self}.RequireGrad().SetGrad(Array{grad});
                              }
                          } else {
                              Array{self}.ClearGrad();
                          }
                      })
        .def_property_readonly("dtype", [](const ArrayBodyPtr& self) { return Array{self}.dtype(); })
        .def_property_readonly("element_bytes", [](const ArrayBodyPtr& self) { return Array{self}.element_bytes(); })
        .def_property_readonly("is_contiguous", [](const ArrayBodyPtr& self) { return Array{self}.is_contiguous(); })
        .def_property_readonly("ndim", [](const ArrayBodyPtr& self) { return Array{self}.ndim(); })
        .def_property_readonly("offset", [](const ArrayBodyPtr& self) { return Array{self}.offset(); })
        .def_property_readonly("shape", [](const ArrayBodyPtr& self) { return Array{self}.shape(); })
        .def_property_readonly("total_bytes", [](const ArrayBodyPtr& self) { return Array{self}.total_bytes(); })
        .def_property_readonly("total_size", [](const ArrayBodyPtr& self) { return Array{self}.total_size(); })
        .def_property_readonly("_debug_data_memory_address",  // These methods starting with `_debug_` are stubs for testing
                               [](const ArrayBodyPtr& self) { return reinterpret_cast<std::uintptr_t>(Array{self}.data().get()); })
        .def_property_readonly("_debug_flat_data", [](const ArrayBodyPtr& self) {
            py::list list;
            Array array{self};

            // Copy data into the list
            VisitDtype(array.dtype(), [&array, &list](auto pt) {
                using T = typename decltype(pt)::type;
                auto size = array.total_size();
                const T& data = *std::static_pointer_cast<const T>(array.data());
                for (int64_t i = 0; i < size; ++i) {
                    list.append((&data)[i]);
                }
            });

            return list;
        });
}

}  // namespace xchainer
