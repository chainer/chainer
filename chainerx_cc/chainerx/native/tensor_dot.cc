#include "chainerx/native/tensor_dot.h"

#include <algorithm>
#include <cstdint>
#include <tuple>

#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/device.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"

namespace chainerx {
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

}  // namespace

Array TensorDot(const Array& a, const Array& b, const Axes& a_axis, const Axes& b_axis, Dtype out_dtype) {
    CHAINERX_ASSERT(a_axis.ndim() == b_axis.ndim());
    CHAINERX_ASSERT(a.ndim() >= a_axis.ndim());
    CHAINERX_ASSERT(b.ndim() >= b_axis.ndim());
    int8_t axis_ndim = a_axis.ndim();

    // Compute the product of reduced dimensions and check that corresponding dimensions in a_axis and b_axis are of equal length.
    int64_t axis_total_size = 1;
    for (int8_t i = 0; i < axis_ndim; ++i) {
        int64_t a_dim = a.shape()[a_axis[i]];
        CHAINERX_ASSERT(a_dim == b.shape()[b_axis[i]]);
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
    Array dot_out = Empty(dot_shape, out_dtype, a.device());
    a.device().backend().CallKernel<DotKernel>(
            a.Transpose(a_roll_axes).Reshape(a_shape), b.Transpose(b_roll_axes).Reshape(b_shape), dot_out);

    // Reshape and return the output array.
    Shape out_shape = a_remain_dims;
    std::copy(b_remain_dims.begin(), b_remain_dims.end(), std::back_inserter(out_shape));
    return dot_out.Reshape(out_shape);
}

}  // namespace native
}  // namespace chainerx
