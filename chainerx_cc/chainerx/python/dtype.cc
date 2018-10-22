#include "chainerx/python/dtype.h"

#include <string>

#include <pybind11/numpy.h>

#include "chainerx/dtype.h"
#include "chainerx/scalar.h"

#include "chainerx/macro.h"
#include "chainerx/python/common.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;  // standard convention

Dtype GetDtypeFromString(const std::string& name) {
    // From Python type names
    if (name == "bool") {
        return Dtype::kBool;
    }
    if (name == "int") {
        // NumPy returns the dtype corresponding to C long.
        // ChainerX does not follow that.
        return Dtype::kInt64;
    }
    if (name == "float") {
        return Dtype::kFloat64;
    }

    // Alias specific to Python binding
    if (name == "bool_") {
        return Dtype::kBool;
    }

    // From Dtype names
    return chainerx::GetDtype(name);
}

Dtype GetDtypeFromNumpyDtype(const py::dtype& npdtype) {
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
    throw DtypeError{"unsupported NumPy dtype"};
}

Dtype GetDtype(py::handle handle) {
    // From chainerx::Dtype
    if (py::isinstance<Dtype>(handle)) {
        return py::cast<Dtype>(handle);
    }

    // From Python types
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyBool_Type)) {  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        return Dtype::kBool;
    }
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type)) {  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        return Dtype::kInt64;
    }
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) {  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        return Dtype::kFloat64;
    }

    // From NumPy dtypes
    if (py::isinstance<py::dtype>(handle)) {
        return GetDtypeFromNumpyDtype(py::cast<py::dtype>(handle));
    }

    // From string
    if (py::isinstance<py::str>(handle)) {
        return GetDtypeFromString(py::cast<std::string>(handle));
    }

    // From NumPy dtype class
    auto numpy_module = py::module::import("numpy");
    if (handle.is(numpy_module.attr("bool_"))) {
        return Dtype::kBool;
    }
    if (handle.is(numpy_module.attr("int8"))) {
        return Dtype::kInt8;
    }
    if (handle.is(numpy_module.attr("int16"))) {
        return Dtype::kInt16;
    }
    if (handle.is(numpy_module.attr("int32"))) {
        return Dtype::kInt32;
    }
    if (handle.is(numpy_module.attr("int64"))) {
        return Dtype::kInt64;
    }
    if (handle.is(numpy_module.attr("uint8"))) {
        return Dtype::kUInt8;
    }
    if (handle.is(numpy_module.attr("float32"))) {
        return Dtype::kFloat32;
    }
    if (handle.is(numpy_module.attr("float64"))) {
        return Dtype::kFloat64;
    }

    throw py::type_error{"Dtype not understood: " + py::cast<std::string>(py::repr(handle))};
}

py::dtype GetNumpyDtype(Dtype dtype) {
    static py::dtype np_bool{"bool"};
    static py::dtype np_int8{"int8"};
    static py::dtype np_int16{"int16"};
    static py::dtype np_int32{"int32"};
    static py::dtype np_int64{"int64"};
    static py::dtype np_uint8{"uint8"};
    static py::dtype np_float32{"float32"};
    static py::dtype np_float64{"float64"};

    switch (dtype) {
        case Dtype::kUInt8:
            return np_uint8;
        case Dtype::kBool:
            return np_bool;
        case Dtype::kInt8:
            return np_int8;
        case Dtype::kInt16:
            return np_int16;
        case Dtype::kInt32:
            return np_int32;
        case Dtype::kInt64:
            return np_int64;
        case Dtype::kFloat32:
            return np_float32;
        case Dtype::kFloat64:
            return np_float64;
        default:
            CHAINERX_NEVER_REACH();
    }
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
