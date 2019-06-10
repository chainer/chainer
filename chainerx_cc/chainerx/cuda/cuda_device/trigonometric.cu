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
#include "chainerx/numeric.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {

namespace {

template <typename T>
struct TanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Tan(x); }
};

}  // namespace

void CudaDevice::Tan(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(TanImpl<T>{}, x_cast, out);
    });
}

namespace {

template <typename T>
struct ArcsinImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Arcsin(x); }
};

}  // namespace

void CudaDevice::Arcsin(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(ArcsinImpl<T>{}, x_cast, out);
    });
}

namespace {

template <typename T>
struct ArccosImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Arccos(x); }
};

}  // namespace

void CudaDevice::Arccos(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(ArccosImpl<T>{}, x_cast, out);
    });
}

namespace {

template <typename T>
struct ArctanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Arctan(x); }
};

}  // namespace

void CudaDevice::Arctan(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(ArctanImpl<T>{}, x_cast, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
