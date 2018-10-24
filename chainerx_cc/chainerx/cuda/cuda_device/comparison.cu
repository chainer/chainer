#include "chainerx/cuda/cuda_device.h"

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
struct EqualImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
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
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 != x2; }
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
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 > x2; }
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
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 >= x2; }
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
    __device__ void operator()(int64_t /*i*/, T x1, bool& out) { out = !x1; }
};

}  // namespace

void CudaDevice::LogicalNot(const Array& x1, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(LogicalNotImpl<T>{}, x1, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
