#include <cstdlib>
#include <cstring>
#include <string>

#include <pybind11/pybind11.h>
#include <gsl/gsl>

#include "chainerx/python/array.h"
#include "chainerx/python/array_index.h"
#include "chainerx/python/backend.h"
#include "chainerx/python/backprop_mode.h"
#include "chainerx/python/backward.h"
#include "chainerx/python/chainer_interop.h"
#include "chainerx/python/check_backward.h"
#include "chainerx/python/common.h"
#include "chainerx/python/context.h"
#include "chainerx/python/cuda/cuda_module.h"
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
    py::options options;
    options.disable_function_signatures();

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
    InitChainerxArray(m);
    InitChainerxBackward(m);
    InitChainerxCheckBackward(m);
    InitChainerxRoutines(m);
    InitChainerxChainerInterop(m);

    m.def("_is_debug", []() -> bool { return CHAINERX_DEBUG; });

    // chainerx.testing (chainerx._testing)
    //
    // Attributes under chainerx._testing are aliased by chainerx.testing in chainerx/testing/__init__.py.
    // This aliasing is needed because chainerx.testing already exists outside the Python binding and we do not want the following
    // sub-module registration to shadow it.
    pybind11::module m_testing = m.def_submodule("_testing");
    testing::testing_internal::InitChainerxTestingModule(m_testing);

    // chainerx._pybind_cuda
    //
    // Define the sub-module only if CUDA is available.
#ifdef CHAINERX_ENABLE_CUDA
    pybind11::module m_cuda = m.def_submodule("_pybind_cuda");
    cuda::cuda_internal::InitChainerxCudaModule(m_cuda);
#endif  // CHAINERX_ENABLE_CUDA

    // Modifies __doc__ property of a pybind-generated function object.
    m.def("_set_pybind_doc", [](py::handle obj, const std::string& docstring) {
        if (!py::isinstance<py::function>(obj)) {
            throw py::type_error{"Object is not a function."};
        }

        // std::malloc should be used here, since pybind uses std::free to free ml_doc.
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        gsl::owner<char*> c_docstring = static_cast<char*>(std::malloc(docstring.size() + 1));
        if (c_docstring == nullptr) {
            return;
        }

        std::strncpy(c_docstring, docstring.c_str(), docstring.size() + 1);

        auto func = py::cast<py::function>(obj);
        auto cfunc = func.cpp_function();
        auto py_cfunc = reinterpret_cast<PyCFunctionObject*>(cfunc.ptr());  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        py_cfunc->m_ml->ml_doc = c_docstring;
    });
}

}  // namespace
}  // namespace python_internal
}  // namespace python
}  // namespace chainerx

PYBIND11_MODULE(_core, m) {  // NOLINT
    chainerx::python::python_internal::InitChainerxModule(m);
}
