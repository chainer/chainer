#include <pybind11/pybind11.h>

#include "xchainer/python/array.h"
#include "xchainer/python/array_index.h"
#include "xchainer/python/backend.h"
#include "xchainer/python/backprop_mode.h"
#include "xchainer/python/backward.h"
#include "xchainer/python/check_backward.h"
#include "xchainer/python/common.h"
#include "xchainer/python/context.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/graph.h"
#include "xchainer/python/routines.h"
#include "xchainer/python/scalar.h"
#include "xchainer/python/testing/testing_module.h"

namespace xchainer {
namespace python {
namespace internal {
namespace {

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    // xchainer
    InitXchainerGraph(m);
    InitXchainerContext(m);
    InitXchainerContextScope(m);
    InitXchainerBackend(m);
    InitXchainerBackpropMode(m);
    InitXchainerDevice(m);
    InitXchainerDeviceScope(m);
    InitXchainerDtype(m);
    InitXchainerError(m);
    InitXchainerScalar(m);
    InitXchainerArrayIndex(m);
    InitXchainerArray(m);
    InitXchainerBackward(m);
    InitXchainerCheckBackward(m);
    InitXchainerRoutines(m);

    // xchainer.testing
    pybind11::module m_testing = m.def_submodule("testing");
    testing::internal::InitXchainerTestingModule(m_testing);
}

}  // namespace
}  // namespace internal
}  // namespace python
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::python::internal::InitXchainerModule(m);
}
