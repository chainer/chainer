#include "chainerx/cuda/cuda_device.h"

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_conv.h"
#include "chainerx/dtype.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

Array CudaDevice::Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        Dtype out_dtype) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(*this);
    return device_internals.cuda_conv().Conv(*this, x, w, b, stride, pad, cover_all, out_dtype);
}

Array CudaDevice::ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size,
        Dtype out_dtype) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(*this);
    return device_internals.cuda_conv().ConvTranspose(*this, x, w, b, stride, pad, out_size, out_dtype);
}

Array CudaDevice::ConvGradWeight(
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(*this);
    return device_internals.cuda_conv().ConvGradWeight(*this, w_dtype, w_shape, x, gy, stride, pad, cover_all);
}

}  // namespace cuda
}  // namespace chainerx
