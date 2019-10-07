#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/cuda/numeric_limits.cuh"
#include "chainerx/cuda/reduce.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/reduction.h"
#include "chainerx/kernels/sorting.h"
#include "chainerx/macro.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/reduction_kernel_arg.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct ArgMaxImpl {
    using CudaType = cuda_internal::DataType<T>;
    struct MaxAndArgMax {
        CudaType max;
        int64_t argmax;
    };
    __device__ MaxAndArgMax Identity() { return {CudaType{}, -1}; }
    __device__ MaxAndArgMax MapIn(CudaType in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
        // Note that `next` can be the return value of `Identity()` in which case `accum` should not be updated.
        if (next.argmax != -1 && (accum.argmax == -1 || accum.max < next.max)) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
};

class CudaArgMaxKernel : public ArgMaxKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Reduce<T, int64_t>(a, axis, out, ArgMaxImpl<T>{});
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ArgMaxKernel, CudaArgMaxKernel);

}  // namespace

namespace {

template <typename T>
struct ArgMinImpl {
    using CudaType = cuda_internal::DataType<T>;
    struct MinAndArgMin {
        CudaType min;
        int64_t argmin;
    };
    __device__ MinAndArgMin Identity() { return {CudaType{}, -1}; }
    __device__ MinAndArgMin MapIn(CudaType in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MinAndArgMin next, MinAndArgMin& accum) {
        // Note that `next` can be the return value of `Identity()` in which case `accum` should not be updated.
        if (next.argmin != -1 && (accum.argmin == -1 || accum.min > next.min)) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MinAndArgMin accum) { return accum.argmin; }
};

class CudaArgMinKernel : public ArgMinKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Reduce<T, int64_t>(a, axis, out, ArgMinImpl<T>{});
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ArgMinKernel, CudaArgMinKernel);

}  // namespace

namespace {

template <typename In, typename Out>
struct SumImpl {
    using InCudaType = cuda_internal::DataType<In>;
    using OutCudaType = cuda_internal::DataType<Out>;
    __device__ OutCudaType Identity() { return OutCudaType{0}; }
    __device__ OutCudaType MapIn(InCudaType in, int64_t /*index*/) { return static_cast<OutCudaType>(in); }
    __device__ void Reduce(OutCudaType next, OutCudaType& accum) { accum += next; }
    __device__ OutCudaType MapOut(OutCudaType accum) { return accum; }
};

class CudaSumKernel : public SumKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};

        auto do_sum = [&a, &axis, &out](auto in_pt, auto out_pt) {
            using In = typename decltype(in_pt)::type;
            using Out = typename decltype(out_pt)::type;
            Reduce<In, Out>(a, axis, out, SumImpl<In, Out>{});
        };

        VisitDtype(out.dtype(), [a_dtype = a.dtype(), &do_sum](auto out_pt) { VisitDtype(a_dtype, do_sum, out_pt); });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SumKernel, CudaSumKernel);

class CudaCumsumKernel : public CumsumKernel {
public:
    void Call(const Array& a, int8_t axis, const Array& out) override {
        Device& device = a.device();
        CHAINERX_ASSERT(a.shape() == out.shape());
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};

        auto do_sum = [&a, &axis, &out](auto in_pt, auto out_pt) {
            using In = typename decltype(in_pt)::type;
            using Out = typename decltype(out_pt)::type;
            Scan<In, Out>(a, axis, out, SumImpl<In, Out>{});
        };

        VisitDtype(out.dtype(), [a_dtype = a.dtype(), &do_sum](auto out_pt) { VisitDtype(a_dtype, do_sum, out_pt); });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CumsumKernel, CudaCumsumKernel);

namespace {

template <typename T>
struct NanArgMaxImpl {
    using CudaType = cuda_internal::DataType<T>;
    struct MaxAndNanArgMax {
        CudaType max;
        int64_t nanargmax;
    };
    __device__ MaxAndNanArgMax Identity() { return {CudaType{}, -1}; }
    __device__ MaxAndNanArgMax MapIn(CudaType in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MaxAndNanArgMax next, MaxAndNanArgMax& accum) {
        // Note that `next` can be the return value of `Identity()` in which case `accum` should not be updated.
        if (next.nanargmax != -1 && (accum.nanargmax == -1 || accum.max < next.max)) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MaxAndNanArgMax accum) { return accum.nanargmax; }
};

class CudaNanArgMaxKernel : public NanArgMaxKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Reduce<T, int64_t>(a, axis, out, NanArgMaxImpl<T>{});
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(NanArgMaxKernel, CudaNanArgMaxKernel);

}  // namespace

namespace {

template <typename T>
struct NanArgMinImpl {
    using CudaType = cuda_internal::DataType<T>;
    struct MinAndNanArgMin {
        CudaType min;
        int64_t nanargmin;
    };
    __device__ MinAndNanArgMin Identity() { return {CudaType{}, -1}; }
    __device__ MinAndNanArgMin MapIn(CudaType in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MinAndNanArgMin next, MinAndNanArgMin& accum) {
        // Note that `next` can be the return value of `Identity()` in which case `accum` should not be updated.
        if (next.nanargmin != -1 && (accum.nanargmin == -1 || accum.min > next.min)) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MinAndNanArgMin accum) { return accum.nanargmin; }
};

class CudaNanArgMinKernel : public NanArgMinKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Reduce<T, int64_t>(a, axis, out, NanArgMinImpl<T>{});
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(NanArgMinKernel, CudaNanArgMinKernel);

}  // namespace

}  // namespace

namespace {

template <typename In, typename Out>
struct NansumImpl {
    using InCudaType = cuda_internal::DataType<In>;
    using OutCudaType = cuda_internal::DataType<Out>;
    __device__ OutCudaType Identity() { return OutCudaType{0}; }
    __device__ OutCudaType MapIn(InCudaType in, int64_t /*index*/) { return static_cast<OutCudaType>(in); }
    __device__ void Reduce(OutCudaType next, OutCudaType& accum) { accum += next; }
    __device__ OutCudaType MapOut(OutCudaType accum) { return accum; }
};

class CudaNansumKernel : public NansumKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};

        auto do_nansum = [&a, &axis, &out](auto in_pt, auto out_pt) {
            using In = typename decltype(in_pt)::type;
            using Out = typename decltype(out_pt)::type;
            Reduce<In, Out>(a, axis, out, NansumImpl<In, Out>{});
        };

        VisitDtype(out.dtype(), [a_dtype = a.dtype(), &do_nansum](auto out_pt) { VisitDtype(a_dtype, do_nansum, out_pt); });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(NansumKernel, CudaNansumKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
