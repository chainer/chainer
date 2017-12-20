#include "xchainer/python/scalar.h"

#include <pybind11/operators.h>

#include "xchainer/scalar.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerScalar(pybind11::module& m) {
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
}
}  // namespace xchainer
