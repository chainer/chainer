#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
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
    CudaSetDeviceScope scope{index()};
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SqrtImpl<T>{}, x, out);
    });
}

namespace {

template <typename T>
__device__ bool IsNan(T /*value*/) {
    return false;
}
__device__ bool IsNan(double value) { return isnan(value); }
__device__ bool IsNan(float value) { return isnan(value); }

template <typename T>
struct IsNanImpl {
    __device__ void operator()(int64_t /*i*/, T x, bool& out) { out = IsNan(x); }
};

}  // namespace

void CudaDevice::IsNan(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(IsNanImpl<T>{}, x, out);
    });
}

namespace {

template <typename T>
__device__ bool IsInf(T /*value*/) {
    return false;
}
__device__ bool IsInf(double value) { return isinf(value); }
__device__ bool IsInf(float value) { return isinf(value); }

template <typename T>
struct IsInfImpl {
    __device__ void operator()(int64_t /*i*/, T x, bool& out) { out = IsInf(x); }
};

}  // namespace

void CudaDevice::IsInf(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(IsInfImpl<T>{}, x, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
