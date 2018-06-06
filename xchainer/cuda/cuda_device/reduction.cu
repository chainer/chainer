#include "xchainer/cuda/cuda_device.h"

#include <cassert>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/reduce.cuh"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/shape.h"

namespace xchainer {
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
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, int64_t>(a, axis, out), ArgMaxImpl<T>{});
    });
}

namespace {

template <typename T>
struct SumImpl {
    __device__ T Identity() { return T{0}; }
    __device__ T MapIn(T in, int64_t /*index*/) { return in; }
    __device__ void Reduce(T next, T& accum) { accum += next; }
    __device__ T MapOut(T accum) { return accum; }
};

}  // namespace

void CudaDevice::Sum(const Array& a, const Axes& axis, const Array& out) {
    assert(xchainer::internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), SumImpl<T>{});
    });
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
    assert(xchainer::internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), AMaxImpl<T>{});
    });
}

}  // namespace cuda
}  // namespace xchainer
