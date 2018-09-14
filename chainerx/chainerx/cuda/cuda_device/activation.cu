#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {

namespace {

template <typename T>
struct IfLessElseASSAImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
    T x2;
    T pos;
};

}  // namespace

void CudaDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(IfLessElseASSAImpl<T>{static_cast<T>(x2), static_cast<T>(pos)}, x1, neg, out);
    });
}

namespace {

template <typename T>
struct TanhImpl {
    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = std::tanh(x); }
};

}  // namespace

void CudaDevice::Tanh(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(TanhImpl<T>{}, x, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
