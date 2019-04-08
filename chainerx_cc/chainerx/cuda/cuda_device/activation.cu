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
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {

namespace {

template <typename In, typename Out>
struct IfLessElseASSAImpl {
    using CudaTypeIn = cuda_internal::DataType<In>;
    using CudaTypeOut = cuda_internal::DataType<Out>;
    __device__ void operator()(int64_t /*i*/, CudaTypeIn x1, CudaTypeOut neg, CudaTypeOut& out) { out = x1 < x2 ? pos : neg; }
    CudaTypeIn x2;
    CudaTypeOut pos;
};

}  // namespace

void CudaDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    Dtype x_dtype = ResultType(x1, x2);
    const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
    const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
    CudaSetDeviceScope scope{index()};
    VisitDtype(x_dtype, [&](auto x_pt) {
        using In = typename decltype(x_pt)::type;
        using CudaTypeIn = cuda_internal::DataType<In>;
        VisitDtype(out.dtype(), [&](auto pt) {
            using Out = typename decltype(pt)::type;
            using CudaTypeOut = cuda_internal::DataType<Out>;
            Elementwise<const In, const Out, Out>(
                    IfLessElseASSAImpl<In, Out>{static_cast<CudaTypeIn>(x2), static_cast<CudaTypeOut>(pos)}, x1_cast, neg_cast, out);
        });
    });
}

namespace {

template <typename In, typename Out>
struct IfGreaterElseASSAImpl {
    using CudaTypeIn = cuda_internal::DataType<In>;
    using CudaTypeOut = cuda_internal::DataType<Out>;
    __device__ void operator()(int64_t /*i*/, CudaTypeIn x1, CudaTypeOut neg, CudaTypeOut& out) { out = x1 > x2 ? pos : neg; }
    CudaTypeIn x2;
    CudaTypeOut pos;
};

}  // namespace

namespace {

template <typename T>
struct IfGreaterElseAAAAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType pos, CudaType neg, CudaType& out) {
        out = x1 > x2 ? pos : neg;
    }
};

}  // namespace

void CudaDevice::IfGreaterElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    Dtype x_dtype = ResultType(x1, x2);
    const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
    const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
    CudaSetDeviceScope scope{index()};
    VisitDtype(x_dtype, [&](auto x_pt) {
        using In = typename decltype(x_pt)::type;
        using CudaTypeIn = cuda_internal::DataType<In>;
        VisitDtype(out.dtype(), [&](auto pt) {
            using Out = typename decltype(pt)::type;
            using CudaTypeOut = cuda_internal::DataType<Out>;
            Elementwise<const In, const Out, Out>(
                    IfGreaterElseASSAImpl<In, Out>{static_cast<CudaTypeIn>(x2), static_cast<CudaTypeOut>(pos)}, x1_cast, neg_cast, out);
        });
    });
}

void CudaDevice::IfGreaterElseAAAA(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, x2, pos, neg, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using CudaType = cuda_internal::DataType<T>;
        Elementwise<const T, const T, const T, const T, T>(IfGreaterElseAAAAImpl<T>{}, x1, x2, pos, neg, out);
    });
}

namespace {

template <typename T>
struct TanhImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Tanh(x); }
};

}  // namespace

void CudaDevice::Tanh(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(TanhImpl<T>{}, x_cast, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
