#include "xchainer/python/error.h"

#include "xchainer/error.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerError(pybind11::module& m) {
    py::register_exception<XchainerError>(m, "XchainerError");
    py::register_exception<DtypeError>(m, "DtypeError");
}
}  // namespace xchainer
