#include "xchainer/python/array.h"

#include <algorithm>

#include <pybind11/numpy.h>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"

namespace xchainer {

namespace py = pybind11;

Dtype ConvertDtype(py::dtype npdtype) {
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

Array InitByShapeDtypeList(const Shape& shape, Dtype dtype, py::list list) {
    auto total_size = shape.total_size();
    auto bytes = GetElementSize(dtype) * total_size;
    if (static_cast<size_t>(total_size) != list.size()) {
        throw DimensionError("Invalid data length");
    }

    std::shared_ptr<void> ptr = std::make_unique<uint8_t[]>(bytes);
    auto func = [&](auto dummy) {
        using T = decltype(dummy);
        std::transform(list.begin(), list.end(), reinterpret_cast<T*>(ptr.get()), [](auto& item) { return py::cast<T>(item); });
    };
    switch (dtype) {
        case Dtype::kBool:
            func(static_cast<bool>(0));
            break;
        case Dtype::kInt8:
            func(static_cast<int8_t>(0));
            break;
        case Dtype::kInt16:
            func(static_cast<int16_t>(0));
            break;
        case Dtype::kInt32:
            func(static_cast<int32_t>(0));
            break;
        case Dtype::kInt64:
            func(static_cast<int64_t>(0));
            break;
        case Dtype::kUInt8:
            func(static_cast<uint8_t>(0));
            break;
        case Dtype::kFloat32:
            func(static_cast<float>(0));
            break;
        case Dtype::kFloat64:
            func(static_cast<double>(0));
            break;
        default:
            assert(0);
    }
    return Array{shape, dtype, ptr};
}

std::unique_ptr<Array> InitByArray(py::array array) {
    if (!(array.flags() & py::array::c_style)) {
        throw DimensionError("cannot convert non-contiguous NumPy array to Array");
    }

    // TODO(hvy): When Unified Memory Array creation and its Python binding is in-place, create the Array on the correct device
    Dtype dtype = ConvertDtype(array.dtype());
    py::buffer_info info = array.request();
    Shape shape(info.shape);
    std::shared_ptr<void> data(std::make_shared<py::array>(std::move(array)), array.mutable_data());

    return std::make_unique<Array>(shape, dtype, data);
}

void InitXchainerArray(pybind11::module& m) {
    py::class_<Array>{m, "Array"}
        .def(py::init(&InitByShapeDtypeList))
        .def(py::init(&InitByArray))
        .def("__repr__", static_cast<std::string (Array::*)() const>(&Array::ToString))
        .def_property_readonly("dtype", &Array::dtype)
        .def_property_readonly("shape", &Array::shape)
        .def_property_readonly("is_contiguous", &Array::is_contiguous)
        .def_property_readonly("total_size", &Array::total_size)
        .def_property_readonly("element_bytes", &Array::element_bytes)
        .def_property_readonly("total_bytes", &Array::total_bytes)
        .def_property_readonly("offset", &Array::offset)
        .def_property_readonly("debug_flat_data",
                               [](const Array& self) {  // This method is a stub for testing
                                   py::list list;
                                   auto size = self.total_size();
                                   auto func = [&](auto dummy) {
                                       using T = decltype(dummy);
                                       const T& data = *std::static_pointer_cast<const T>(self.data());
                                       for (int64_t i = 0; i < size; ++i) {
                                           list.append((&data)[i]);
                                       }
                                   };
                                   switch (self.dtype()) {
                                       case Dtype::kBool:
                                           func(static_cast<bool>(0));
                                           break;
                                       case Dtype::kInt8:
                                           func(static_cast<int8_t>(0));
                                           break;
                                       case Dtype::kInt16:
                                           func(static_cast<int16_t>(0));
                                           break;
                                       case Dtype::kInt32:
                                           func(static_cast<int32_t>(0));
                                           break;
                                       case Dtype::kInt64:
                                           func(static_cast<int64_t>(0));
                                           break;
                                       case Dtype::kUInt8:
                                           func(static_cast<uint8_t>(0));
                                           break;
                                       case Dtype::kFloat32:
                                           func(static_cast<float>(0));
                                           break;
                                       case Dtype::kFloat64:
                                           func(static_cast<double>(0));
                                           break;
                                       default:
                                           assert(0);
                                   }
                                   return list;
                               })
        .def("__add__", static_cast<Array (Array::*)(const Array&) const>(&Array::Add))
        .def("__iadd__", static_cast<Array& (Array::*)(const Array&)>(&Array::IAdd))
        .def("__mul__", static_cast<Array (Array::*)(const Array&) const>(&Array::Mul))
        .def("__imul__", static_cast<Array& (Array::*)(const Array&)>(&Array::IMul));
}

}  // namespace xchainer
