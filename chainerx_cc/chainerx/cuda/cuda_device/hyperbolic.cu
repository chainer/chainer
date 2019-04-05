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

<<<<<<< HEAD
template <typename CudaType, typename Op>
struct UnaryOpImpl {
    Op op;

    UnaryOpImpl(Op op) : op{op} {}

    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = op(x); }
=======
template <typename T>
struct CudaUnaryOp {
    T (*func)(T);

    CudaUnaryOp(T (*func)(T)) : func{func} {}

    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = func(x); }
>>>>>>> add-a/sinh-a/cosh
};

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
<<<<<<< HEAD
            auto op = cuda::Sinh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
=======
            Elementwise<const T, T>(CudaUnaryOp<CudaType>{cuda::Sinh}, x_cast, out);
>>>>>>> add-a/sinh-a/cosh
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
<<<<<<< HEAD
            auto op = cuda::Cosh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
=======
            Elementwise<const T, T>(CudaUnaryOp<CudaType>{cuda::Cosh}, x_cast, out);
>>>>>>> add-a/sinh-a/cosh
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
<<<<<<< HEAD
            auto op = cuda::Arcsinh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
=======
            Elementwise<const T, T>(CudaUnaryOp<CudaType>{cuda::Arcsinh}, x_cast, out);
>>>>>>> add-a/sinh-a/cosh
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
<<<<<<< HEAD
            auto op = cuda::Arccosh<CudaType>{};
            auto functor = UnaryOpImpl<CudaType, decltype(op)>{op};
            Elementwise<const T, T>(functor, x_cast, out);
=======
            Elementwise<const T, T>(CudaUnaryOp<CudaType>{cuda::Arccosh}, x_cast, out);
>>>>>>> add-a/sinh-a/cosh
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(ArccoshOp, CudaArccoshOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
