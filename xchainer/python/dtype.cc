#include "xchainer/python/dtype.h"

#include <string>

#include "xchainer/dtype.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

void InitXchainerDtype(pybind11::module& m) {
    py::enum_<Dtype> e{m, "Dtype"};
    for (Dtype dtype : GetAllDtypes()) {
        e.value(GetDtypeName(dtype), dtype);
    }
    e.export_values();
    e.def(py::init(&GetDtype));
    e.def_property_readonly("char", [](Dtype dtype) { return std::string(1, GetCharCode(dtype)); });
    e.def_property_readonly("itemsize", &GetElementSize);
    e.def_property_readonly("name", &GetDtypeName);
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
