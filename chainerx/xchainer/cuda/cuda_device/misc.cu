#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"

namespace chainerx {
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
}  // namespace chainerx
