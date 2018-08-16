#include "xchainer/cuda/cuda_device.h"

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
struct EqualImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
};

}  // namespace

void CudaDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(EqualImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct GreaterImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 > x2; }
};

}  // namespace

void CudaDevice::Greater(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(GreaterImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct NotImpl {
    __device__ void operator()(int64_t /*i*/, T x1, bool& out) { out = !x1; }
};

}  // namespace

void CudaDevice::Not(const Array& x1, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(NotImpl<T>{}, x1, out);
    });
}

}  // namespace cuda
}  // namespace xchainer
