#include "xchainer/python/dtype.h"

#include <string>

#include "xchainer/dtype.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

void InitXchainerDtype(pybind11::module& m) {
    py::enum_<Dtype> e{m, "dtype"};
    for (Dtype dtype : GetAllDtypes()) {
        e.value(GetDtypeName(dtype), dtype);
    }
    e.export_values();
    e.def(py::init(&GetDtype));
    e.def(py::init([](Dtype dtype) { return dtype; }));
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
