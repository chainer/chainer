#include <string>

#include "chainerx/python/scalar.h"

#include <pybind11/operators.h>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/python/array.h"
#include "chainerx/python/py_cached_objects.h"
#include "chainerx/scalar.h"

#include "chainerx/python/common.h"
#include "chainerx/python/dtype.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

namespace {

Scalar HandleToScalar(py::handle obj) {
    if (py::isinstance<Scalar>(obj)) {
        return py::cast<Scalar>(obj);
    }
    if (py::isinstance<py::float_>(obj)) {
        return Scalar{py::cast<double>(obj)};
    }
    if (py::isinstance<py::bool_>(obj)) {
        return Scalar{py::cast<bool>(obj)};
    }
    if (py::isinstance<py::int_>(obj)) {
        return Scalar{py::cast<int64_t>(obj)};
    }
    if (py::isinstance(obj, python_internal::GetCachedNumpyNumber()) || py::isinstance(obj, python_internal::GetCachedNumpyBool())) {
        py::object sc = obj.attr("tolist")();  // Do not pass it as a temporary
        return HandleToScalar(sc);
    }
    throw py::type_error{"Cannot interpret as a scalar: " + py::cast<std::string>(py::repr(obj.attr("__class__")))};
}

}  // namespace

void InitChainerxScalar(pybind11::module& m) {
    // This binding allows implicit casting from Python and NumPy scalars.
    py::class_<Scalar> c{m, "_Scalar"};
    c.def(py::init([](py::handle obj) { return HandleToScalar(obj); }));
    py::implicitly_convertible<py::handle, Scalar>();
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
