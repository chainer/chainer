#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/math.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename CudaType, typename Op>
struct UnaryOpImpl {
    Op op;

    explicit UnaryOpImpl(Op op) : op{op} {}

    __device__ inline void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = op(x); }
};

template <typename T>
struct SinImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Sin(x); }
};

class CudaSinOp : public SinOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(SinImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(SinOp, CudaSinOp);

template <typename T>
struct CosImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Cos(x); }
};

class CudaCosOp : public CosOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(CosImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(CosOp, CudaCosOp);

template <typename T>
struct TanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Tan(x); }
};

class CudaTanOp : public TanOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(TanImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(TanOp, CudaTanOp);

template <typename T>
struct ArcsinImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Arcsin(x); }
};

class CudaArcsinOp : public ArcsinOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(ArcsinImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(ArcsinOp, CudaArcsinOp);

template <typename T>
struct ArccosImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Arccos(x); }
};

class CudaArccosOp : public ArccosOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(ArccosImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(ArccosOp, CudaArccosOp);

template <typename T>
struct ArctanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Arctan(x); }
};

class CudaArctanOp : public ArctanOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(ArctanImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(ArctanOp, CudaArctanOp);

class CudaSinhOp : public SinhOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            auto op = cuda::Sinh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(SinhOp, CudaSinhOp);

class CudaCoshOp : public CoshOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            auto op = cuda::Cosh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(CoshOp, CudaCoshOp);

class CudaArcsinhOp : public ArcsinhOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            auto op = cuda::Arcsinh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(ArcsinhOp, CudaArcsinhOp);

class CudaArccoshOp : public ArccoshOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            auto op = cuda::Arccosh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(ArccoshOp, CudaArccoshOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
