#include "xchainer/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/elementwise.cuh"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace cuda {

namespace {

template <typename T>
struct FillImpl {
    __device__ void operator()(int64_t /*i*/, T& out) { out = value; }
    T value;
};

}  // namespace

void CudaDevice::Fill(const Array& out, Scalar value) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(FillImpl<T>{static_cast<T>(value)}, out);
    });
}

}  // namespace cuda
}  // namespace xchainer
