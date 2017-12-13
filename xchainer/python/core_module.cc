#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <gsl/gsl>
#include <iterator>
#include <sstream>
#include <string>
#include "xchainer/dtype.h"
#include "xchainer/error.h"

namespace xchainer {
namespace {

namespace py = pybind11;  // standard convention

Dtype ConvertDtype(py::dtype npdtype) {
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
    throw DtypeError("unsupported numpy dtype");
}

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "Xchainer";
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
    }

    m.def("array", []() {});
}

}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) { xchainer::InitXchainerModule(m); }
