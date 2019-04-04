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
    // This binding allows implicit casting from `py::bool_`, `py::int_` and `py::float_`.
    py::class_<Scalar> c{m, "_Scalar"};
    c.def(py::init<bool>());
    c.def(py::init<int64_t>());
    c.def(py::init<double>());
    py::implicitly_convertible<py::bool_, Scalar>();
    py::implicitly_convertible<py::int_, Scalar>();
    py::implicitly_convertible<py::float_, Scalar>();
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
