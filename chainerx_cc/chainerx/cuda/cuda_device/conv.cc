#include "chainerx/cuda/cuda_device.h"

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_conv.h"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/dims.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {
namespace {

class CudaConvKernel : public ConvKernel {
public:
    Array Call(
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            Dtype out_dtype,
            const absl::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_conv().Conv(device, x, w, b, stride, pad, cover_all, out_dtype);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ConvKernel, CudaConvKernel);

class CudaConvTransposeKernel : public ConvTransposeKernel {
public:
    Array Call(
            const Array& x,
            const Array& w,
            const absl::optional<Array>& b,
            const Dims& stride,
            const Dims& pad,
            const Dims& out_size,
            Dtype out_dtype,
            const absl::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }
        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_conv().ConvTranspose(device, x, w, b, stride, pad, out_size, out_dtype);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ConvTransposeKernel, CudaConvTransposeKernel);

class CudaConvGradWeightKernel : public ConvGradWeightKernel {
public:
    Array Call(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const Dims& stride,
            const Dims& pad,
            bool cover_all,
            const absl::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }
        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_conv().ConvGradWeight(device, w_dtype, w_shape, x, gy, stride, pad, cover_all);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ConvGradWeightKernel, CudaConvGradWeightKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
