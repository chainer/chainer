#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/reduce.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/reduction_kernel_arg.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct ArgMaxImpl {
    struct MaxAndArgMax {
        T max;
        int64_t argmax;
    };
    __device__ MaxAndArgMax Identity() { return {T{}, -1}; }
    __device__ MaxAndArgMax MapIn(T in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
        if (accum.argmax < 0 || accum.max < next.max) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
};

}  // namespace

void CudaDevice::ArgMax(const Array& a, const Axes& axis, const Array& out) {
    CheckDevicesCompatible(a, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce<T, int64_t>(a, axis, out, ArgMaxImpl<T>{});
    });
}

namespace {

template <typename In, typename Out>
struct SumImpl {
    __device__ Out Identity() { return Out{0}; }
    __device__ Out MapIn(In in, int64_t /*index*/) { return static_cast<Out>(in); }
    __device__ void Reduce(Out next, Out& accum) { accum += next; }
    __device__ Out MapOut(Out accum) { return accum; }
};

}  // namespace

void CudaDevice::Sum(const Array& a, const Axes& axis, const Array& out) {
    CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CudaSetDeviceScope scope{index()};

    auto do_sum = [&a, &axis, &out](auto in_pt, auto out_pt) {
        using In = typename decltype(in_pt)::type;
        using Out = typename decltype(out_pt)::type;
        Reduce<In, Out>(a, axis, out, SumImpl<In, Out>{});
    };

    VisitDtype(out.dtype(), [a_dtype = a.dtype(), &do_sum](auto out_pt) { VisitDtype(a_dtype, do_sum, out_pt); });
}

namespace {
template <typename T>
__device__ bool IsNan(T /*value*/) {
    return false;
}
__device__ bool IsNan(double value) { return ::isnan(value); }
__device__ bool IsNan(float value) { return ::isnan(value); }

template <typename T>
struct AMaxImpl {
    __device__ T Identity() { return NumericLimits<T>::LowestOrInf(); }
    __device__ T MapIn(T in, int64_t /*index*/) { return in; }
    __device__ void Reduce(T next, T& accum) {
        if (IsNan(next) || accum < next) {
            accum = next;
        }
    }
    __device__ T MapOut(T accum) { return accum; }
};
}  // namespace

void CudaDevice::AMax(const Array& a, const Axes& axis, const Array& out) {
    CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce<T, T>(a, axis, out, AMaxImpl<T>{});
    });
}

}  // namespace cuda
}  // namespace chainerx
