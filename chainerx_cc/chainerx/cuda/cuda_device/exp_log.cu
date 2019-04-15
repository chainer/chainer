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
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct ExpImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Exp(x); }
};

class CudaExpOp : public ExpOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&x_cast, &out](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(ExpImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(ExpOp, CudaExpOp);

template <typename T>
struct LogImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Log(x); }
};

class CudaLogOp : public LogOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&x_cast, &out](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(LogImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(LogOp, CudaLogOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
