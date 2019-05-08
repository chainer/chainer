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
struct PowImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = cuda::Pow(x1, x2); }
};

class CudaPowKernel : public PowKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) {
        x1.device().CheckDevicesCompatible(x1, x2, out);
        CudaSetDeviceScope scope{x1.device().index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, T>(PowImpl<T>{}, x1, x2, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(PowKernel, CudaPowKernel);

template <typename T>
struct PowASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = cuda::Pow(x1, x2); }
    CudaType x2;
};

class CudaPowASKernel : public PowASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) {
        x1.device().CheckDevicesCompatible(x1, out);
        CudaSetDeviceScope scope{x1.device().index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(PowASImpl<T>{static_cast<CudaType>(x2)}, x1, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(PowASKernel, CudaPowASKernel);

template <typename T>
struct PowSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = cuda::Pow(x1, x2); }
    CudaType x1;
};

class CudaPowSAKernel : public PowSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) {
        x2.device().CheckDevicesCompatible(x2, out);
        CudaSetDeviceScope scope{x2.device().index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(PowSAImpl<T>{static_cast<CudaType>(x1)}, x2, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(PowSAKernel, CudaPowSAKernel);

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
struct IsFiniteImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = !(cuda::IsInf(x) || cuda::IsNan(x)); }
};

class CudaIsFiniteKernel : public IsFiniteKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, bool>(IsFiniteImpl<T>{}, x, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(IsFiniteKernel, CudaIsFiniteKernel);

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
