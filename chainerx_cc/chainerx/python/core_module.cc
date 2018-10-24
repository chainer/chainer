#include <pybind11/pybind11.h>

#include "chainerx/python/array.h"
#include "chainerx/python/array_index.h"
#include "chainerx/python/backend.h"
#include "chainerx/python/backprop_mode.h"
#include "chainerx/python/backward.h"
#include "chainerx/python/chainer_interop.h"
#include "chainerx/python/check_backward.h"
#include "chainerx/python/common.h"
#include "chainerx/python/context.h"
#include "chainerx/python/device.h"
#include "chainerx/python/dtype.h"
#include "chainerx/python/error.h"
#include "chainerx/python/graph.h"
#include "chainerx/python/routines.h"
#include "chainerx/python/scalar.h"
#include "chainerx/python/testing/testing_module.h"

namespace chainerx {
namespace python {
namespace python_internal {
namespace {

void InitChainerxModule(pybind11::module& m) {
    m.doc() = "ChainerX";
    m.attr("__name__") = "chainerx";  // Show each member as "chainerx.*" instead of "chainerx.core.*"

    // chainerx
    InitChainerxContext(m);
    InitChainerxContextScope(m);
    InitChainerxGraph(m);
    InitChainerxBackpropScope(m);
    InitChainerxBackend(m);
    InitChainerxBackpropMode(m);
    InitChainerxDevice(m);
    InitChainerxDeviceScope(m);
    InitChainerxDtype(m);
    InitChainerxError(m);
    InitChainerxScalar(m);
    InitChainerxArrayIndex(m);
    InitChainerxArray(m);
    InitChainerxBackward(m);
    InitChainerxCheckBackward(m);
    InitChainerxRoutines(m);
    InitChainerxChainerInterop(m);

    // chainerx.testing (chainerx._testing)
    //
    // Attributes under chainerx._testing are aliased by chainerx.testing in chainerx/testing/__init__.py.
    // This aliasing is needed because chainerx.testing already exists outside the Python binding and we do not want the following
    // sub-module registration to shadow it.
    pybind11::module m_testing = m.def_submodule("_testing");
    testing::testing_internal::InitChainerxTestingModule(m_testing);
}

}  // namespace
}  // namespace python_internal
}  // namespace python
}  // namespace chainerx

PYBIND11_MODULE(_core, m) {  // NOLINT
    chainerx::python::python_internal::InitChainerxModule(m);
}
