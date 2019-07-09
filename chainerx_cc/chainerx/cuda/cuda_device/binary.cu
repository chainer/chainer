#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/binary.h"
#include "chainerx/routines/binary.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(BitwiseAndKernel, { out = x1 & x2; }, VisitIntegralDtype);

template <typename T>
struct BitwiseAndASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = x1 & x2; }
    CudaType x2;
};

class CudaBitwiseAndASKernel : public BitwiseAndASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(BitwiseAndASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(BitwiseAndASKernel, CudaBitwiseAndASKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(BitwiseOrKernel, { out = x1 | x2; }, VisitIntegralDtype);

template <typename T>
struct BitwiseOrASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = x1 | x2; }
    CudaType x2;
};

class CudaBitwiseOrASKernel : public BitwiseOrASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(BitwiseOrASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(BitwiseOrASKernel, CudaBitwiseOrASKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(BitwiseXorKernel, { out = x1 ^ x2; }, VisitIntegralDtype);

template <typename T>
struct BitwiseXorASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = x1 ^ x2; }
    CudaType x2;
};

class CudaBitwiseXorASKernel : public BitwiseXorASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(BitwiseXorASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(BitwiseXorASKernel, CudaBitwiseXorASKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(LeftShiftAAKernel, { out = x1 << x2; }, VisitShiftDtype);

template <typename T>
struct LeftShiftASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = x1 << x2; }
    CudaType x2;
};

class CudaLeftShiftASKernel : public LeftShiftASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(LeftShiftASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(LeftShiftASKernel, CudaLeftShiftASKernel);

template <typename T>
struct LeftShiftSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = x1 << x2; }
    CudaType x1;
};

class CudaLeftShiftSAKernel : public LeftShiftSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(LeftShiftSAImpl<T>{static_cast<CudaType>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(LeftShiftSAKernel, CudaLeftShiftSAKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(RightShiftAAKernel, { out = x1 >> x2; }, VisitShiftDtype);

template <typename T>
struct RightShiftASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = x1 >> x2; }
    CudaType x2;
};

class CudaRightShiftASKernel : public RightShiftASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(RightShiftASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RightShiftASKernel, CudaRightShiftASKernel);

template <typename T>
struct RightShiftSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = x1 >> x2; }
    CudaType x1;
};

class CudaRightShiftSAKernel : public RightShiftSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitIntegralDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(RightShiftSAImpl<T>{static_cast<CudaType>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RightShiftSAKernel, CudaRightShiftSAKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
