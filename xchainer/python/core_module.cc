#include <pybind11/pybind11.h>

#include "xchainer/array.h"

#include "xchainer/python/array.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/type_caster.h"  // need to include in every compilation unit of the Python extension module
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/python/cuda/hello.h"
#endif  // XCHAINER_ENABLE_CUDA

namespace xchainer {
namespace {

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"
}
}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::InitXchainerModule(m);
    xchainer::InitXchainerDevice(m);
    xchainer::InitXchainerDtype(m);
    xchainer::InitXchainerError(m);
    xchainer::InitXchainerScalar(m);
    xchainer::InitXchainerShape(m);
    xchainer::InitXchainerArray(m);
#ifdef XCHAINER_ENABLE_CUDA
    xchainer::cuda::InitXchainerCudaHello(m);
#endif  // XCHAINER_ENABLE_CUDA
}
