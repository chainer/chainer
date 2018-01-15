#include "xchainer/python/scalar.h"

#include <pybind11/operators.h>

#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerScalar(pybind11::module& m) {
    py::class_<Scalar>(m, "Scalar")
        .def(py::init<bool>())
        .def(py::init<int64_t>())
        .def(py::init<double>())
        .def(py::init<bool, Dtype>())
        .def(py::init<int64_t, Dtype>())
        .def(py::init<double, Dtype>())
        .def(py::self == py::self)
        .def("__eq__", [](Scalar scalar, bool value) { return scalar == Scalar{value}; })
        .def("__eq__", [](Scalar scalar, int64_t value) { return scalar == Scalar{value}; })
        .def("__eq__", [](Scalar scalar, double value) { return scalar == Scalar{value}; })
        .def(+py::self)
        .def(-py::self)
        .def("__bool__", &Scalar::operator bool)
        .def("__int__", &Scalar::operator int64_t)
        .def("__float__", &Scalar::operator double)
        .def("__repr__", &Scalar::ToString)
        .def_property_readonly("dtype", &Scalar::dtype)
        .def("tolist", [](Scalar scalar) -> py::object {
            switch (GetKind(scalar.dtype())) {
                case DtypeKind::kBool:
                    return VisitDtype(scalar.dtype(), [scalar](auto pt) -> py::object {
                        using T = typename decltype(pt)::type;
                        return py::bool_{static_cast<T>(scalar)};
                    });
                case DtypeKind::kInt:  // fallthrough
                case DtypeKind::kUInt:
                    return VisitDtype(scalar.dtype(), [scalar](auto pt) -> py::object {
                        using T = typename decltype(pt)::type;
                        return py::int_{static_cast<int64_t>(static_cast<T>(scalar))};
                    });
                case DtypeKind::kFloat:
                    return VisitDtype(scalar.dtype(), [scalar](auto pt) -> py::object {
                        using T = typename decltype(pt)::type;
                        return py::float_{static_cast<double>(static_cast<T>(scalar))};
                    });
                default:
                    assert(false);  // should never be reached
            }
            return {};
        });

    py::implicitly_convertible<py::bool_, Scalar>();
    py::implicitly_convertible<py::int_, Scalar>();
    py::implicitly_convertible<py::float_, Scalar>();
}
}  // namespace xchainer
