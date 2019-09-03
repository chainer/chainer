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
#include "chainerx/kernels/misc.h"
#include "chainerx/routines/type_util.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(SqrtKernel, { out = cuda::Sqrt(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(SquareKernel, { out = x * x; }, VisitNumericDtype);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(AbsKernel, { out = cuda::Abs(x); }, VisitNumericDtype);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(SignKernel, { out = cuda::Sign(x); }, VisitNumericDtype);

template <typename In, typename Out>
struct IfLessElseASSAImpl {
    using InCudaType = cuda_internal::DataType<In>;
    using OutCudaType = cuda_internal::DataType<Out>;
    __device__ void operator()(int64_t /*i*/, InCudaType x1, OutCudaType neg, OutCudaType& out) { out = x1 < x2 ? pos : neg; }
    InCudaType x2;
    OutCudaType pos;
};

class CudaIfLessElseASSAKernel : public IfLessElseASSAKernel {
public:
    void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            using InCudaType = cuda_internal::DataType<In>;
            VisitNumericDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                using OutCudaType = cuda_internal::DataType<Out>;
                Elementwise<const In, const Out, Out>(
                        IfLessElseASSAImpl<In, Out>{static_cast<InCudaType>(x2), static_cast<OutCudaType>(pos)}, x1_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(IfLessElseASSAKernel, CudaIfLessElseASSAKernel);

template <typename In, typename Out>
struct IfGreaterElseASSAImpl {
    using InCudaType = cuda_internal::DataType<In>;
    using OutCudaType = cuda_internal::DataType<Out>;
    __device__ void operator()(int64_t /*i*/, InCudaType x1, OutCudaType neg, OutCudaType& out) { out = x1 > x2 ? pos : neg; }
    InCudaType x2;
    OutCudaType pos;
};

class CudaIfGreaterElseASSAKernel : public IfGreaterElseASSAKernel {
public:
    void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            using InCudaType = cuda_internal::DataType<In>;
            VisitNumericDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                using OutCudaType = cuda_internal::DataType<Out>;
                Elementwise<const In, const Out, Out>(
                        IfGreaterElseASSAImpl<In, Out>{static_cast<InCudaType>(x2), static_cast<OutCudaType>(pos)}, x1_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(IfGreaterElseASSAKernel, CudaIfGreaterElseASSAKernel);

template <typename In, typename Out>
struct IfGreaterElseAAAAImpl {
    using InCudaType = cuda_internal::DataType<In>;
    using OutCudaType = cuda_internal::DataType<Out>;
    __device__ void operator()(int64_t /*i*/, InCudaType x1, InCudaType x2, OutCudaType pos, OutCudaType neg, OutCudaType& out) {
        out = x1 > x2 ? pos : neg;
    }
};

class CudaIfGreaterElseAAAAKernel : public IfGreaterElseAAAAKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, pos, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& x2_cast = x2.dtype() == x_dtype ? x2 : x2.AsType(x_dtype);
        const Array& pos_cast = pos.dtype() == out.dtype() ? pos : pos.AsType(out.dtype());
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            VisitNumericDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                Elementwise<const In, const In, const Out, const Out, Out>(
                        IfGreaterElseAAAAImpl<In, Out>{}, x1_cast, x2_cast, pos_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(IfGreaterElseAAAAKernel, CudaIfGreaterElseAAAAKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
