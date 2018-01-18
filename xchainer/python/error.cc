#include "xchainer/python/error.h"

#include "xchainer/error.h"

#include "xchainer/python/type_caster.h"  // need to include in every compilation unit of the Python extension module

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerError(pybind11::module& m) {
    py::register_exception<XchainerError>(m, "XchainerError");
    py::register_exception<DeviceError>(m, "DeviceError");
    py::register_exception<DtypeError>(m, "DtypeError");
    py::register_exception<DimensionError>(m, "DimensionError");
}

}  // namespace xchainer
