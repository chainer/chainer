#include <pybind11/pybind11.h>

#include "xchainer/constant.h"
#include "xchainer/python/array.h"
#include "xchainer/python/array_index.h"
#include "xchainer/python/backend.h"
#include "xchainer/python/backward.h"
#include "xchainer/python/check_backward.h"
#include "xchainer/python/common.h"
#include "xchainer/python/context.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"

namespace xchainer {
namespace python {
namespace internal {
namespace {

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    m.attr("DEFAULT_GRAPH_ID") = kDefaultGraphId;

    InitXchainerContext(m);
    InitXchainerBackend(m);
    InitXchainerDevice(m);
    InitXchainerDtype(m);
    InitXchainerError(m);
    InitXchainerScalar(m);
    InitXchainerArrayIndex(m);
    InitXchainerArray(m);
    InitXchainerBackward(m);
    InitXchainerCheckBackward(m);
}

}  // namespace
}  // namespace internal
}  // namespace python
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::python::internal::InitXchainerModule(m);
}
