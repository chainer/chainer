#include <pybind11/pybind11.h>

#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"

namespace xchainer {
namespace {

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"
}
}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {
    xchainer::InitXchainerModule(m);
    xchainer::InitXchainerDtype(m);
    xchainer::InitXchainerError(m);
    xchainer::InitXchainerScalar(m);
}
