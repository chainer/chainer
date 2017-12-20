#include "xchainer/python/dtype.h"

#include "xchainer/dtype.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerDtype(pybind11::module& m) {
    py::enum_<Dtype> dtype_type(m, "Dtype");
    for (Dtype dtype : GetAllDtypes()) {
        dtype_type.value(GetDtypeName(dtype), dtype);
    }
    dtype_type.export_values();
    dtype_type.def(py::init(&GetDtype));
    dtype_type.def_property_readonly("char", [](Dtype dtype) { return std::string(1, GetCharCode(dtype)); });
    dtype_type.def_property_readonly("itemsize", &GetElementSize);
    dtype_type.def_property_readonly("name", &GetDtypeName);
}
}  // namespace xchainer
