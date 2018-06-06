#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/axes.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/native/reduce.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/math.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace native {

namespace {

Array Im2Col(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    auto ndim = static_cast<int8_t>(kernel_size.size());  // Number of input image dimensions.
    assert(ndim == static_cast<int8_t>(stride.size()));
    assert(ndim == static_cast<int8_t>(pad.size()));
    assert(ndim + 2 == x.ndim());  // Batch and channel dimensions.

    Device& device = x.device();

    // Create a padded copy of the input image.
    // TODO(hvy): Use the Pad function when implemented.
    Shape padded_shape = x.shape();
    std::vector<ArrayIndex> unpadded_slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int64_t i = 0; i < ndim; ++i) {
        padded_shape[i + 2] += pad[i] * 2 + (cover_all ? stride[i] - 1 : 0);  // Pad on both sides.
        unpadded_slice.emplace_back(Slice{pad[i], pad[i] + x.shape()[i]});
    }
    // TODO(hvy): Allow non-zero padding.
    Array padded_x = Zeros(padded_shape, x.dtype(), device);
    device.Copy(x, padded_x.At(unpadded_slice));

    // Create the output array.
    StackVector<int64_t, kMaxNdim> out_dims;  // Number of patches along each axis
    for (int8_t i = 0; i < ndim; ++i) {
        out_dims.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], kernel_size[i], stride[i], pad[i], cover_all));
        assert(out_dims.back() > 0);
    }

    int64_t batch_size = x.shape()[0];
    int64_t channels = x.shape()[1];

    Shape out_shape{batch_size, channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(out_shape));
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
    Array out = Empty(out_shape, x.dtype(), device);

    // Write to the output array.
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};
        Indexer<> kernel_indexer{Shape{kernel_size.begin(), kernel_size.end()}};
        Indexer<> out_dims_indexer{Shape{out_dims.begin(), out_dims.end()}};
        Indexer<> x_indexer{padded_x.shape()};
        Indexer<> out_indexer{out.shape()};
        IndexableArray<const T> x_iarray{padded_x};
        IndexableArray<T> out_iarray{out};

        // Indices over input image.
        NdimIndex img_index{ndim};

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_out_dims = out_dims_indexer.It(0); it_out_dims; ++it_out_dims) {
                for (int i = 0; i < ndim; ++i) {
                    img_index.index()[i] = it_out_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
                    auto it_x = x_indexer.At(it_batch_channel, img_index);
                    auto it_out = out_indexer.At(it_batch_channel, it_kernel, it_out_dims);

                    // Write the output column value.
                    out_iarray[it_out] = x_iarray[it_x];
                }
            }
        }
    });

    return out;
}

// Returns necessary data for TensorDot for one of the input arrays.
// It is called for both inputs to TensorDot.
//
// It returns a tuple of:
// 0. Permuted axes for transpose, moving axes to be reduced to either front or back of array axes.
// 1. Non-reduced shape dimensions to be used in the output shape of TensorDot.
std::tuple<Axes, Shape> GetTensorDotRollAxes(const Shape& shape, const Axes& reduce_axes, bool reduced_axes_first) {
    bool to_reduce[kMaxNdim]{};  // Initialized with false.
    Shape remain_dims;
    Axes roll_axes;
    for (int8_t i = 0; i < reduce_axes.ndim(); ++i) {
        gsl::at(to_reduce, reduce_axes[i]) = true;
    }

    // There are two steps:
    // A. Insert axes to be reduced to roll_axes.
    // B. Insert non-reduced axes to roll_axes.
    // The order of these steps depends on reduced_axes_first.
    for (int step = 0; step < 2; ++step) {
        if ((step == 0) == reduced_axes_first) {
            // Step A.
            for (int8_t i = 0; i < shape.ndim(); ++i) {
                if (gsl::at(to_reduce, i)) {
                    roll_axes.emplace_back(i);
                }
            }
        } else {
            // Step B.
            for (int8_t i = 0; i < shape.ndim(); ++i) {
                if (!gsl::at(to_reduce, i)) {
                    roll_axes.emplace_back(i);
                    remain_dims.emplace_back(shape[i]);
                }
            }
        }
    }

    return std::make_tuple(roll_axes, remain_dims);
}

Array TensorDot(const Array& a, const Array& b, const Axes& a_axis, const Axes& b_axis) {
    assert(a_axis.ndim() == b_axis.ndim());
    assert(a.ndim() >= a_axis.ndim());
    assert(b.ndim() >= b_axis.ndim());
    int8_t axis_ndim = a_axis.ndim();

    // Compute the product of reduced dimensions and check that corresponding dimensions in a_axis and b_axis are of equal length.
    int64_t axis_total_size = 1;
    for (int8_t i = 0; i < axis_ndim; ++i) {
        int64_t a_dim = a.shape()[a_axis[i]];
        assert(a_dim == b.shape()[b_axis[i]]);
        axis_total_size *= a_dim;
    }

    // Compute necessary data for Dot and Reshape.
    auto a_tup = GetTensorDotRollAxes(a.shape(), a_axis, false);
    auto b_tup = GetTensorDotRollAxes(b.shape(), b_axis, true);
    const Axes& a_roll_axes = std::get<0>(a_tup);
    const Axes& b_roll_axes = std::get<0>(b_tup);
    const Shape& a_remain_dims = std::get<1>(a_tup);
    const Shape& b_remain_dims = std::get<1>(b_tup);
    int64_t a_remain_total_size = a_remain_dims.GetTotalSize();
    int64_t b_remain_total_size = b_remain_dims.GetTotalSize();
    Shape a_shape{a_remain_total_size, axis_total_size};
    Shape b_shape{axis_total_size, b_remain_total_size};

    // Compute the dot product between a and b reshaped to 2-dimensions.
    Shape dot_shape{a_remain_total_size, b_remain_total_size};
    Array dot_out = Empty(dot_shape, a.dtype(), a.device());
    a.device().Dot(a.Transpose(a_roll_axes).Reshape(a_shape), b.Transpose(b_roll_axes).Reshape(b_shape), dot_out);

    // Reshape and return the output array.
    Shape out_shape = a_remain_dims;
    std::copy(b_remain_dims.begin(), b_remain_dims.end(), std::back_inserter(out_shape));
    return dot_out.Reshape(out_shape);
}

}  // namespace

Array NativeDevice::Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    int8_t ndim = w.ndim() - 2;  // Number of spacial dimensions

    // Compute the kernel size from the weight array.
    StackVector<int64_t, kMaxNdim> kernel_size;
    std::copy_n(w.shape().begin() + 2, ndim, std::back_inserter(kernel_size));

    // Convert to colum representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
    Array col = Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all);

    // Compute the tensor dot product of col and w, reducing (channel, k_1, k_2, ..., k_n).
    Axes axes;
    axes.resize(ndim + 1);
    std::iota(axes.begin(), axes.end(), 1);
    Array y = TensorDot(col, w.AsConstant(), axes, axes);  // (batch_size, out_1, out_2, ..., out_n, out_channel)

    // Add bias, if given.
    if (b.has_value()) {
        y += b->AsConstant();
    }

    // Move the out channel axis to the second
    Axes roll_axes;
    roll_axes.resize(y.ndim());
    roll_axes[0] = 0;
    roll_axes[1] = ndim + 1;
    std::iota(roll_axes.begin() + 2, roll_axes.end(), 1);
    Array out = y.Transpose(roll_axes);

    return out;
}

Array NativeDevice::ConvGradWeight(
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    assert(x.ndim() == w_shape.ndim());
    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions

    // Compute the kernel size
    StackVector<int64_t, kMaxNdim> kernel_size{w_shape.begin() + 2, w_shape.end()};

    // Im2Col
    Array col = Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all);

    // TensorDot
    Axes out_axes{0};
    Axes col_axes{0};
    for (int8_t i = 0; i < ndim; ++i) {
        out_axes.emplace_back(int64_t{2 + i});
        col_axes.emplace_back(int64_t{2 + ndim + i});
    }
    return TensorDot(gy.AsConstant(), col, out_axes, col_axes).AsType(w_dtype, false);
}

namespace {

Array Col2Im(
        const Array& col,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    // Cannot use const due to internal compiler error with gcc 5.4.0.
    int8_t batch_size = col.shape()[0];
    int8_t channels = col.shape()[1];
    auto ndim = static_cast<int8_t>(stride.size());

    Shape padded_shape{batch_size, channels};
    for (int8_t i = 0; i < ndim; ++i) {
        padded_shape.emplace_back(out_size[i] + 2 * pad[i] + stride[i] - 1);
    }
    Array padded_out = Zeros(padded_shape, col.dtype(), col.device());

    // Write to the output array
    VisitDtype(col.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};
        Indexer<> kernel_indexer{Shape{col.shape().begin() + 2, col.shape().begin() + 2 + ndim}};
        Indexer<> in_image_dims_indexer{Shape{col.shape().begin() + 2 + ndim, col.shape().end()}};
        Indexer<> col_indexer{col.shape()};
        Indexer<> padded_out_indexer{padded_shape};
        IndexableArray<const T> col_iarray{col};
        IndexableArray<T> padded_out_iarray{padded_out};

        // Indices over the output image.
        NdimIndex out_image_index{ndim};

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_in_image_dims = in_image_dims_indexer.It(0); it_in_image_dims; ++it_in_image_dims) {
                for (int8_t i = 0; i < ndim; ++i) {
                    out_image_index.index()[i] = it_in_image_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
                    auto it_col = col_indexer.At(it_batch_channel, it_kernel, it_in_image_dims);
                    auto it_padded_out = padded_out_indexer.At(it_batch_channel, out_image_index);
                    padded_out_iarray[it_padded_out] += col_iarray[it_col];
                }
            }
        }
    });

    std::vector<ArrayIndex> slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int8_t i = 0; i < ndim; ++i) {
        slice.emplace_back(Slice{pad[i], pad[i] + out_size[i]});
    }
    return padded_out.At(slice);
}

}  // namespace

Array NativeDevice::ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    Array col = TensorDot(w.AsConstant(), x.AsConstant(), {0}, {1});  // shape: out_channel, k_1, ..., k_n, batch_size, out_1, ..., out_n
    col = RollAxis(col, x.ndim() - 1);  // batch axis is rolled to the top

    Array y = Col2Im(col, stride, pad, out_size);  // shape: batch_size, out_channel, out_size...

    // Add bias, if given.
    if (b.has_value()) {
        std::vector<ArrayIndex> slice{NewAxis{}, Slice{}};
        for (size_t i = 0; i < out_size.size(); ++i) {
            slice.emplace_back(NewAxis{});
        }
        y += b->AsConstant().At(slice);
    }

    return y;
}

namespace {

Array ExpandDims(const Array& a, const Axes& axes) {
    // Compute the new set of strides with the new axes.
    int8_t expanded_ndim = a.ndim() + axes.ndim();
    int8_t i_axis = 0;
    int8_t i_stride = 0;
    const Strides& strides = a.strides();
    Strides expanded_strides;
    for (int8_t i = 0; i < expanded_ndim; ++i) {
        if (i_axis < axes.ndim() && i == axes[i_axis]) {
            expanded_strides.emplace_back(int64_t{1});
            ++i_axis;
        } else {
            expanded_strides.emplace_back(strides[i_stride]);
            ++i_stride;
        }
    }
    assert(i_axis == axes.ndim());
    assert(i_stride == strides.ndim());
    assert(expanded_strides.ndim() == a.ndim() + axes.ndim());

    return xchainer::internal::MakeArray(
            xchainer::internal::ExpandShape(a.shape(), axes), expanded_strides, a.dtype(), a.device(), a.data(), a.offset());
}

void Mean(const Array& a, const Axes& axis, const Array& out) {
    Device& device = a.device();
    device.Sum(a, axis, out);
    device.DivideAS(out, xchainer::internal::CountItemsAlongAxes(a.shape(), axis), out);
}

void Var(const Array& a, const Array& mean, const Axes& axis, const Array& out) {
    Array out_pre_reduction = EmptyLike(a, a.device());
    VisitDtype(out.dtype(), [&a, &mean, &out_pre_reduction](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T a, T mean, T& out) {
                T diff = a - mean;
                out = diff * diff;
            }
        };
        Elementwise<const T, const T, T>(Impl{}, a, mean.BroadcastTo(a.shape()), out_pre_reduction);
    });
    Mean(out_pre_reduction, axis, out);
}

class NativeBatchNormForwardBackward : public xchainer::BatchNormForwardBackward {
public:
    Array Forward(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis) override {
        Dtype dtype = x.dtype();

        Array x_mean = xchainer::internal::EmptyReduced(x.shape(), dtype, axis, true, x.device());
        Mean(x, axis, x_mean);

        Array x_var = xchainer::internal::EmptyReduced(x.shape(), dtype, axis, true, x.device());
        Var(x, x_mean, axis, x_var);

        Array out = EmptyLike(x, x.device());
        int64_t n = x.GetTotalSize() / gamma.GetTotalSize();
        VisitFloatingPointDtype(
                dtype, [&x, &x_mean, &x_var, &running_mean, &running_var, &gamma, &beta, eps, decay, &axis, &out, n](auto pt) {
                    using T = typename decltype(pt)::type;

                    // Compute the batch normalization.
                    struct BatchNormImpl {
                        void operator()(int64_t /*i*/, T x, T x_mean, T x_var, T gamma, T beta, T& out) {
                            out = (x - x_mean) / std::sqrt(x_var + eps) * gamma + beta;
                        }
                        T eps;
                    };
                    Elementwise<const T, const T, const T, const T, const T, T>(
                            BatchNormImpl{static_cast<T>(eps)},
                            x,
                            x_mean.BroadcastTo(out.shape()),
                            x_var.BroadcastTo(out.shape()),
                            ExpandDims(gamma, axis).BroadcastTo(out.shape()),
                            ExpandDims(beta, axis).BroadcastTo(out.shape()),
                            out);

                    // Update the running mean and variance in-place using an unbiased estimate.
                    struct UpdateStatsImpl {
                        void operator()(int64_t /*i*/, T mean, T var, T& running_mean, T& running_var) {
                            running_mean *= decay;
                            running_mean += (T{1} - decay) * mean;
                            running_var *= decay;
                            running_var += (T{1} - decay) * adjust * var;
                        }
                        T decay;
                        T adjust;
                    };
                    Elementwise<const T, const T, T, T>(
                            UpdateStatsImpl{static_cast<T>(decay), static_cast<T>(n) / std::max(n - 1, int64_t{1})},
                            x_mean,
                            x_var,
                            ExpandDims(running_mean, axis),
                            ExpandDims(running_var, axis));
                });
        return out;
    }

    std::array<Array, 3> Backward(
            const Array& /*x*/, const Array& /*gamma*/, const Array& /*gout*/, Scalar /*eps*/, Scalar /*decay*/, const Axes& /*axis*/)
            override {
        return {Array{}, Array{}, Array{}};
    }
};

}  // namespace

std::shared_ptr<BatchNormForwardBackward> NativeDevice::GetBatchNormForwardBackward() {
    return std::make_shared<NativeBatchNormForwardBackward>();
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
