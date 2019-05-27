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
#include "chainerx/kernels/statistics.h"
#include "chainerx/macro.h"
#include "chainerx/numeric_limits.h"
#include "chainerx/reduction_kernel_arg.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct AMaxImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ CudaType Identity() { return cuda::NumericLimits<CudaType>::LowestOrInf(); }
    __device__ CudaType MapIn(CudaType in, int64_t /*index*/) { return in; }
    __device__ void Reduce(CudaType next, CudaType& accum) {
        if (cuda::IsNan(next) || accum < next) {
            accum = next;
        }
    }
    __device__ CudaType MapOut(CudaType accum) { return accum; }
};

class CudaAMaxKernel : public AMaxKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Reduce<T, T>(a, axis, out, AMaxImpl<T>{});
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(AMaxKernel, CudaAMaxKernel);

template <typename T>
struct AMinImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ CudaType Identity() { return cuda::NumericLimits<CudaType>::MaxOrInf(); }
    __device__ CudaType MapIn(CudaType in, int64_t /*index*/) { return in; }
    __device__ void Reduce(CudaType next, CudaType& accum) {
        if (cuda::IsNan(next) || accum > next) {
            accum = next;
        }
    }
    __device__ CudaType MapOut(CudaType accum) { return accum; }
};

class CudaAMinKernel : public AMinKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        Device& device = a.device();
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Reduce<T, T>(a, axis, out, AMinImpl<T>{});
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(AMinKernel, CudaAMinKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
