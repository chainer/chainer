#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct EqualImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 == x2; }
};

}  // namespace

void CudaDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(EqualImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct NotEqualImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 != x2; }
};

}  // namespace

void CudaDevice::NotEqual(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(NotEqualImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct GreaterImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 > x2; }
};

}  // namespace

void CudaDevice::Greater(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(GreaterImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct GreaterEqualImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, bool& out) { out = x1 >= x2; }
};

}  // namespace

void CudaDevice::GreaterEqual(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(GreaterEqualImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct LogicalNotImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = !x; }
};

}  // namespace

void CudaDevice::LogicalNot(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(LogicalNotImpl<T>{}, x, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
