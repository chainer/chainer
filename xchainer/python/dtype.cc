#include "xchainer/python/dtype.h"

#include <string>

#include "xchainer/dtype.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

Dtype GetDtypeFromString(const std::string& name) {
    // From Python type names
    if (name == "bool") {
        return Dtype::kBool;
    }
    if (name == "int") {
        // NumPy returns the dtype corresponding to C long.
        // xChainer does not follow that.
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
    return xchainer::GetDtype(name);
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
    throw DtypeError("unsupported NumPy dtype");
}

Dtype GetDtype(py::handle handle) {
    // From xchainer::Dtype
    if (py::isinstance<Dtype>(handle)) {
        return py::cast<Dtype>(handle);
    }

    // From Python types
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyBool_Type)) {
        return Dtype::kBool;
    }
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type)) {
        return Dtype::kInt64;
    }
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) {
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

    // TODO(niboshi): Generate richer error message
    throw py::type_error();
}

void InitXchainerDtype(pybind11::module& m) {
    py::enum_<Dtype> e{m, "dtype"};
    for (Dtype dtype : GetAllDtypes()) {
        e.value(dtype == Dtype::kBool ? "bool_" : GetDtypeName(dtype), dtype);
    }
    e.export_values();
    e.def(py::init(&internal::GetDtype));
    e.def_property_readonly("char", [](Dtype dtype) { return std::string(1, GetCharCode(dtype)); });
    e.def_property_readonly("itemsize", &GetElementSize);
    e.def_property_readonly("name", &GetDtypeName);
    e.def("__eq__", [](Dtype self, py::handle other) {
        (void)self;   // unused
        (void)other;  // unused
        return false;
    });
    e.def("__ne__", [](Dtype self, py::handle other) {
        (void)self;   // unused
        (void)other;  // unused
        return true;
    });
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
