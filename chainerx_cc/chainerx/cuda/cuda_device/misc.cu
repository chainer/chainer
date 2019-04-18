#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/math.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct SquareImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = x * x; }
};

class CudaSquareKernel : public SquareKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(SquareImpl<T>{}, x, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SquareKernel, CudaSquareKernel);

template <typename T>
struct SqrtImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Sqrt(x); }
};

class CudaSqrtKernel : public SqrtKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(SqrtImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SqrtKernel, CudaSqrtKernel);

template <typename T>
struct IsNanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = cuda::IsNan(x); }
};

class CudaIsNanKernel : public IsNanKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, bool>(IsNanImpl<T>{}, x, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(IsNanKernel, CudaIsNanKernel);

template <typename T>
struct IsInfImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = cuda::IsInf(x); }
};

class CudaIsInfKernel : public IsInfKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, bool>(IsInfImpl<T>{}, x, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(IsInfKernel, CudaIsInfKernel);

template <typename T>
struct CeilImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Ceil(x); }
};

class CudaCeilKernel : public CeilKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(CeilImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CeilKernel, CudaCeilKernel);

template <typename T>
struct FloorImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Floor(x); }
};

class CudaFloorKernel : public FloorKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(FloorImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(FloorKernel, CudaFloorKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
