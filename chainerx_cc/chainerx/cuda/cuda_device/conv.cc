#include "chainerx/cuda/cuda_device.h"

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_conv.h"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/routines/connection.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace {

class CudaConvOp : public ConvOp {
public:
    Array Call(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            Dtype out_dtype,
            const nonstd::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }

        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_conv().Conv(device, x, w, b, stride, pad, cover_all, out_dtype);
    }
};

CHAINERX_CUDA_REGISTER_OP(ConvOp, CudaConvOp);

class CudaConvTransposeOp : public ConvTransposeOp {
public:
    Array Call(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& out_size,
            Dtype out_dtype,
            const nonstd::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }
        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_conv().ConvTranspose(device, x, w, b, stride, pad, out_size, out_dtype);
    }
};

CHAINERX_CUDA_REGISTER_OP(ConvTransposeOp, CudaConvTransposeOp);

class CudaConvGradWeightOp : public ConvGradWeightOp {
public:
    Array Call(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            const nonstd::optional<Array>& out) override {
        // TODO(niboshi): Implement and test the `out` argument.
        if (out.has_value()) {
            throw NotImplementedError{"Passing out as an argument is not yet supported."};
        }
        CudaDevice& device = dynamic_cast<CudaDevice&>(x.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_conv().ConvGradWeight(device, w_dtype, w_shape, x, gy, stride, pad, cover_all);
    }
};

CHAINERX_CUDA_REGISTER_OP(ConvGradWeightOp, CudaConvGradWeightOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
