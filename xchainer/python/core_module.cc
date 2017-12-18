#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gsl/gsl>

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
        .def("__bool__", &Scalar::UnwrapAndCast<bool>)
        .def("__int__", &Scalar::UnwrapAndCast<int64_t>)
        .def("__float__", &Scalar::UnwrapAndCast<double>)
        .def("__repr__", &Scalar::ToString)
        .def_property_readonly("dtype", &Scalar::dtype);
}

}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) { xchainer::InitXchainerModule(m); }
