#include "xchainer/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/elementwise.cuh"
#include "xchainer/device.h"
#include "xchainer/dtype.h"

namespace xchainer {
namespace cuda {
namespace {

template <typename T>
struct SqrtImpl {
    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = std::sqrt(x); }
};

}  // namespace

void CudaDevice::Sqrt(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SqrtImpl<T>{}, x, out);
    });
}

}  // namespace cuda
}  // namespace xchainer
