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
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    CudaSetDeviceScope scope{index()};
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SqrtImpl<T>{}, x_cast, out);
    });
}

namespace {

template <typename T>
struct PowImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = cuda::Pow(x1, x2); }
};

}  // namespace

void CudaDevice::Pow(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(PowImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct PowASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = cuda::Pow(x1, x2); }
    CudaType x2;
};

}  // namespace

void CudaDevice::PowAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using CudaType = cuda_internal::DataType<T>;
        Elementwise<const T, T>(PowASImpl<T>{static_cast<CudaType>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct PowSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = cuda::Pow(x1, x2); }
    CudaType x1;
};

}  // namespace

void CudaDevice::PowSA(Scalar x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x2, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using CudaType = cuda_internal::DataType<T>;
        Elementwise<const T, T>(PowSAImpl<T>{static_cast<CudaType>(x1)}, x2, out);
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
