#include "xchainer/python/scalar.h"

#include <pybind11/operators.h>

#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

void InitXchainerScalar(pybind11::module& m) {
    py::class_<Scalar> cls{m, "Scalar"};
    cls.def(py::init<bool>());
    cls.def(py::init<int64_t>());
    cls.def(py::init<double>());
    cls.def(py::init<bool, Dtype>());
    cls.def(py::init<int64_t, Dtype>());
    cls.def(py::init<double, Dtype>());
    cls.def(py::self == py::self);  // NOLINT
    cls.def("__eq__", [](Scalar scalar, bool value) { return scalar == Scalar{value}; });
    cls.def("__eq__", [](Scalar scalar, int64_t value) { return scalar == Scalar{value}; });
    cls.def("__eq__", [](Scalar scalar, double value) { return scalar == Scalar{value}; });
    cls.def(+py::self);
    cls.def(-py::self);
    cls.def("__bool__", &Scalar::operator bool);
    cls.def("__int__", &Scalar::operator int64_t);
    cls.def("__float__", &Scalar::operator double);
    cls.def("__repr__", &Scalar::ToString);
    cls.def_property_readonly("dtype", &Scalar::dtype);
    cls.def("tolist", [](Scalar scalar) -> py::object {
        switch (GetKind(scalar.dtype())) {
            case DtypeKind::kBool:
                return py::bool_{static_cast<bool>(scalar)};
            case DtypeKind::kInt:  // fallthrough
            case DtypeKind::kUInt:
                return py::int_{static_cast<int64_t>(scalar)};
            case DtypeKind::kFloat:
                return py::float_{static_cast<double>(scalar)};
            default:
                assert(false);  // should never be reached
        }
        return {};
    });

    py::implicitly_convertible<py::bool_, Scalar>();
    py::implicitly_convertible<py::int_, Scalar>();
    py::implicitly_convertible<py::float_, Scalar>();
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
