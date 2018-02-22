#include "xchainer/python/backend.h"

#include "xchainer/backend.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/native_backend.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;  // standard convention

void InitXchainerBackend(pybind11::module& m) {
    py::class_<Backend> backend(m, "Backend");
    backend.def("get_device", &Backend::GetDevice, py::return_value_policy::reference).def_property_readonly("name", &Backend::GetName);
    py::class_<NativeBackend>(m, "NativeBackend", backend).def(py::init());
#ifdef XCHAINER_ENABLE_CUDA
    py::class_<cuda::CudaBackend>(m, "CudaBackend", backend).def(py::init());
#endif  // XCHAINER_ENABLE_CUDA
}

}  // namespace xchainer
