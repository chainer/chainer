#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct SqrtImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Sqrt(x); }
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
struct IsNanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = cuda::IsNan(x); }
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
struct IsInfImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = cuda::IsInf(x); }
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
