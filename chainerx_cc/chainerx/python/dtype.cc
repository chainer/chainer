#include "chainerx/python/common_export.h"

#include "chainerx/python/dtype.h"

#include <string>

#include <pybind11/numpy.h>

#include "chainerx/dtype.h"
#include "chainerx/scalar.h"

#include "chainerx/macro.h"
#include "chainerx/python/common.h"
#include "chainerx/python/py_cached_objects.h"

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
                case 2:
                    return Dtype::kFloat16;
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
    auto numpy = GetCachedNumpyModule();
    if (handle.is(numpy.attr("bool_"))) {
        return Dtype::kBool;
    }
    if (handle.is(numpy.attr("int8"))) {
        return Dtype::kInt8;
    }
    if (handle.is(numpy.attr("int16"))) {
        return Dtype::kInt16;
    }
    if (handle.is(numpy.attr("int32"))) {
        return Dtype::kInt32;
    }
    if (handle.is(numpy.attr("int64"))) {
        return Dtype::kInt64;
    }
    if (handle.is(numpy.attr("uint8"))) {
        return Dtype::kUInt8;
    }
    if (handle.is(numpy.attr("float16"))) {
        return Dtype::kFloat16;
    }
    if (handle.is(numpy.attr("float32"))) {
        return Dtype::kFloat32;
    }
    if (handle.is(numpy.attr("float64"))) {
        return Dtype::kFloat64;
    }

    throw py::type_error{"Dtype not understood: " + py::cast<std::string>(py::repr(handle))};
}

py::object GetNumpyDtypeFromModule(const py::module& m, Dtype dtype) {
    switch (dtype) {
        case Dtype::kBool:
            return m.attr("_bool");
        case Dtype::kInt8:
            return m.attr("_int8");
        case Dtype::kInt16:
            return m.attr("_int16");
        case Dtype::kInt32:
            return m.attr("_int32");
        case Dtype::kInt64:
            return m.attr("_int64");
        case Dtype::kUInt8:
            return m.attr("_uint8");
        case Dtype::kFloat16:
            return m.attr("_float16");
        case Dtype::kFloat32:
            return m.attr("_float32");
        case Dtype::kFloat64:
            return m.attr("_float64");
        default:
            CHAINERX_NEVER_REACH();
    }
}

void InitChainerxDtype(py::module& m) {
    // Store cached py::dtype objects directly in the core module to optimize the ChainerX dtype to NumPy dtype conversions.
    // This improves the performance of e.g. accessing ndarray.dtype.
    m.attr("_bool") = py::dtype{"bool"};
    m.attr("_int8") = py::dtype{"int8"};
    m.attr("_int16") = py::dtype{"int16"};
    m.attr("_int32") = py::dtype{"int32"};
    m.attr("_int64") = py::dtype{"int64"};
    m.attr("_uint8") = py::dtype{"uint8"};
    m.attr("_float16") = py::dtype{"float16"};
    m.attr("_float32") = py::dtype{"float32"};
    m.attr("_float64") = py::dtype{"float64"};
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
