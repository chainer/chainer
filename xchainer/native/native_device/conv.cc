#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/im2col.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace {

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
    Array col = internal::Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all, 0);

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
    Array col = internal::Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all, 0);

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

}  // namespace native
}  // namespace xchainer
