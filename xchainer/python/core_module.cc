#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace {

namespace py = pybind11;  // standard convention

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    py::register_exception<XchainerError>(m, "XchainerError");
    py::register_exception<DtypeError>(m, "DtypeError");

    //
    // Types
    //

    {
        py::enum_<Dtype> dtype_type(m, "Dtype");
        for (Dtype dtype : GetAllDtypes()) {
            dtype_type.value(GetDtypeName(dtype), dtype);
        }
        dtype_type.export_values();
        dtype_type.def(py::init(&GetDtype));
        dtype_type.def_property_readonly("char", [](Dtype dtype) { return std::string(1, GetCharCode(dtype)); });
        dtype_type.def_property_readonly("itemsize", &GetElementSize);
        dtype_type.def_property_readonly("name", &GetDtypeName);
    }

    py::class_<Scalar>(m, "Scalar")
        .def(py::init<bool>())
        .def(py::init<int64_t>())
        .def(py::init<double>())
        .def(+py::self)
        .def(-py::self)
        .def("__bool__", &Scalar::operator bool)
        .def("__int__", &Scalar::operator int64_t)
        .def("__float__", &Scalar::operator double)
        .def("__repr__", &Scalar::ToString)
        .def_property_readonly("dtype", &Scalar::dtype);

    py::class_<Shape>{m, "Shape"}
        .def(py::init([](py::tuple tup) {  // __init__ by a tuple
            std::vector<int64_t> v;
            std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
            return Shape(v);
        }))
        .def(py::self == py::self)
        .def("__eq__",  // Equality with a tuple
             [](const Shape& self, const py::tuple& tup) {
                 if (static_cast<size_t>(self.ndim()) != tup.size()) {
                     return false;
                 }
                 try {
                     return std::equal(self.begin(), self.end(), tup.begin(), tup.end(), [](const auto& dim, const auto& item) {
                         int64_t dim2 = py::cast<int64_t>(item);
                         return dim == dim2;
                     });
                 } catch (const py::cast_error& e) {
                     return false;
                 }
             })
        .def_property_readonly("ndim", &Shape::ndim)
        .def_property_readonly("size", &Shape::size)
        .def_property_readonly("total_size", &Shape::total_size);

    py::implicitly_convertible<py::tuple, Shape>();

    py::class_<Array, std::shared_ptr<Array>>{m, "Array"}
        .def(py::init([](const Shape& shape, Dtype dtype, py::list list) {
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
        }))
        .def_property_readonly("dtype", &Array::dtype)
        .def_property_readonly("shape", &Array::shape)
        .def_property_readonly("is_contiguous", &Array::is_contiguous)
        .def_property_readonly("total_size", &Array::total_size)
        .def_property_readonly("element_bytes", &Array::element_bytes)
        .def_property_readonly("total_bytes", &Array::total_bytes)
        .def_property_readonly("offset", &Array::offset)
        .def_property_readonly("debug_flat_data", [](const Array& self) {  // This method is a stub for testing
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
        });
}

}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) { xchainer::InitXchainerModule(m); }
