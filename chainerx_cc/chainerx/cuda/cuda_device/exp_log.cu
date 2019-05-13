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
#include "chainerx/kernels/math.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(ExpKernel, { out = cuda::Exp(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(LogKernel, { out = cuda::Log(x); });

CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(Log10Kernel, { out = cuda::Log10(x); });

template <typename T>
struct Log2Impl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Log2(x); }
};

class CudaLog2Kernel : public Log2Kernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&x_cast, &out](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(Log2Impl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(Log2Kernel, CudaLog2Kernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
