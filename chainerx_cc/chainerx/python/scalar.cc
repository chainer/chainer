#include <string>

#include "chainerx/python/scalar.h"

#include <pybind11/operators.h>

#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/scalar.h"

#include "chainerx/python/common.h"
#include "chainerx/python/dtype.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

void InitChainerxScalar(pybind11::module& m) {
    py::class_<Scalar> c{m, "Scalar"};
    c.def(py::init<bool>());
    c.def(py::init<int64_t>());
    c.def(py::init<double>());
    c.def(py::self == py::self);  // NOLINT
    c.def("__eq__", [](Scalar scalar, bool value) { return scalar == Scalar{value}; });
    c.def("__eq__", [](Scalar scalar, int64_t value) { return scalar == Scalar{value}; });
    c.def("__eq__", [](Scalar scalar, double value) { return scalar == Scalar{value}; });
    c.def(+py::self);
    c.def(-py::self);
    c.def("__bool__", &Scalar::operator bool);
    c.def("__int__", &Scalar::operator int64_t);
    c.def("__float__", &Scalar::operator double);
    c.def("__repr__", &Scalar::ToString);
    c.def("tolist", [](Scalar scalar) -> py::object {
        switch (scalar.kind()) {
            case DtypeKind::kBool:
                return py::bool_{static_cast<bool>(scalar)};
            case DtypeKind::kInt:
                return py::int_{static_cast<int64_t>(scalar)};
            case DtypeKind::kFloat:
                return py::float_{static_cast<double>(scalar)};
            default:
                CHAINERX_NEVER_REACH();
        }
        return {};
    });

    py::implicitly_convertible<py::bool_, Scalar>();
    py::implicitly_convertible<py::int_, Scalar>();
    py::implicitly_convertible<py::float_, Scalar>();
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
