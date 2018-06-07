#include "xchainer/cuda/cuda_device.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {

namespace {

void ConvCheckDtype(const Array& x, const Array& w, const nonstd::optional<Array>& b) {
    // TODO(sonots): Support float16
    if (x.dtype() != Dtype::kFloat32 && x.dtype() != Dtype::kFloat64) {
        throw XchainerError{"XChainer cuDNN supports only float32 or float64 arrays, but the input array dtype is: ", x.dtype()};
    }
    if (w.dtype() != x.dtype()) {
        throw XchainerError{"XChainer cuDNN requires the filter (kernel) array dtype: ",
                            w.dtype(),
                            " and the input array dtype: ",
                            x.dtype(),
                            " are same"};
    }
    if (b && b->dtype() != x.dtype()) {
        throw XchainerError{
                "XChainer cuDNN requires the bias array dtype: ", b->dtype(), " and the input array dtype: ", x.dtype(), " are same"};
    }
}

}  // namespace

Array CudaDevice::Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    if (cover_all) {
        throw XchainerError{"CUDA convolution does not support cover_all"};
    }
    if (b) {
        CheckDevicesCompatible(x, w, *b);
    } else {
        CheckDevicesCompatible(x, w);
    }
    ConvCheckDtype(x, w, b);

    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spacial dimensions to be greater than or equal to 2"};
    }
    assert(w.ndim() == x.ndim());
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));

    // w.shape = (out_channels, _, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[0];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    Shape out_shape{batch_size, out_channels};
    for (int8_t i = 0; i < ndim; ++i) {
        out_shape.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], w.shape()[i + 2], stride[i], pad[i], cover_all));
        assert(out_shape.back() > 0);
    }
    Array y = Empty(out_shape, x.dtype(), *this);

    cudnn_context_.ConvolutionForward(x, w, b, y, pad, stride, nonstd::nullopt, 1);

    return y;
}

Array CudaDevice::ConvGradWeight(
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    if (cover_all) {
        throw XchainerError{"CUDA convolution does not support cover_all"};
    }

    CheckDevicesCompatible(x, gy);

    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spacial dimensions to be greater than or equal to 2"};
    }
    assert(x.ndim() == w_shape.ndim());
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));
    assert(gy.ndim() == w_shape.ndim());

    Array gw = Empty(w_shape, w_dtype, *this);
    cudnn_context_.ConvolutionBackwardFilter(x, gy, gw, pad, stride, nonstd::nullopt /*dilation*/, 1 /*groups*/);

    return gw;
}

Array CudaDevice::ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    if (b) {
        CheckDevicesCompatible(x, w, *b);
    } else {
        CheckDevicesCompatible(x, w);
    }
    ConvCheckDtype(x, w, b);

    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spacial dimensions to be greater than or equal to 2"};
    }
    assert(w.ndim() == x.ndim());
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));
    assert(out_size.size() == static_cast<size_t>(ndim));

    // w.shape = (in_channels, out_channels, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[1];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    // (Note that cover_all is not supported in cuDNN implementation.)
    Shape out_shape{batch_size, out_channels};
    std::copy(out_size.begin(), out_size.end(), std::back_inserter(out_shape));

    Array y = Empty(out_shape, x.dtype(), *this);

    cudnn_context_.ConvolutionBackwardData(w, x, b, y, pad, stride, nonstd::nullopt /*dilation*/, 1 /*group*/);

    return y;
}

}  // namespace cuda
}  // namespace xchainer
