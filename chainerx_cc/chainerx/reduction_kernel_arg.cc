#include "chainerx/reduction_kernel_arg.h"

#include <cstdint>
#include <tuple>

#include "chainerx/array.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"
#include "chainerx/squash_dims.h"
#include "chainerx/strides.h"

namespace chainerx {

ReductionArg::ReductionArg(const Array& in, const Axes& axis, const Array& out) : in_{in}, out_{out} {
    Permute(axis);
    Squash();
}

void ReductionArg::Permute(const Axes& axis) {
    // True if some axes are reduced but kept in output as 1-dim axes.
    // Corresponding to keepdim argument in Array::Sum().
    bool has_kept_dims = out_.ndim() + static_cast<int64_t>(axis.size()) != in_.ndim();

    // Prepare axis mappings
    Axes out_axis_map{};  // Mapping from effective output indices to actual output indices
    // (Here "effective output indices" means source indices minus reduction indices.)

    // Example (in the case of has_kept_dims=false):
    // - in_.shape():     (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out_.shape():     (12, 14, 16)
    // - out_axis_map:     (0, 1, 2)
    // - out_shape_:       (12, 14, 16)
    // Example (in the case of has_kept_dims=true):
    // - in_.shape():     (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out_.shape():     (12, 1, 14, 1, 16)
    // - out_axis_map:     (0, 2, 4)
    // - out_shape_:       (12, 14, 16)

    if (has_kept_dims) {
        for (int8_t i : axis) {
            if (out_.shape()[i] != 1) {
                out_axis_map.emplace_back(i);
            }
        }
    }
    {
        size_t i_axis = 0;
        size_t i_out_axis = 0;
        for (int8_t i = 0; i < in_.shape().ndim(); ++i) {
            if (i_axis < axis.size() && i == axis[i_axis]) {
                // i is to be reduced
                ++i_axis;
                if (has_kept_dims) {
                    ++i_out_axis;
                }
            } else {
                // i is not to be reduced
                int64_t out_dim = out_.shape()[i_out_axis];
                if (out_dim != 1) {
                    out_axis_map.emplace_back(static_cast<int8_t>(i_out_axis));
                }
                ++i_out_axis;
            }
        }
        CHAINERX_ASSERT(i_out_axis == out_.shape().size());
        CHAINERX_ASSERT(i_axis == axis.size());
    }
    // Inequality because 1-dim axes are eliminated.
    CHAINERX_ASSERT(out_axis_map.size() <= in_.shape().size());

    // Calculate source axis permutation
    // - in_.shape():     (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - axis_permutes:    (1, 3, 0, 2, 4)
    // - in_shape_:        (13, 15, 12, 14, 16)
    Axes axis_permutes{};
    for (int8_t i : axis) {
        if (in_.shape()[i] != 1) {
            axis_permutes.emplace_back(i);
        }
    }
    {
        size_t i_reduce = 0;
        for (int8_t i = 0; i < in_.ndim(); ++i) {
            if (i_reduce < axis.size() && i == axis[i_reduce]) {
                ++i_reduce;
            } else {
                if (in_.shape()[i] != 1) {
                    axis_permutes.emplace_back(i);
                }
            }
        }
    }
    CHAINERX_ASSERT(axis_permutes.size() <= in_.shape().size());  // Inequality because 1-dim axes are eliminated.

    // 1-dim axes must be eliminated
    CHAINERX_ASSERT(std::find(in_shape_.begin(), in_shape_.end(), 1) == in_shape_.end());
    CHAINERX_ASSERT(std::find(out_shape_.begin(), out_shape_.end(), 1) == out_shape_.end());

    in_shape_ = in_.shape().Permute(axis_permutes);
    in_strides_ = in_.strides().Permute(axis_permutes);
    out_shape_ = out_.shape().Permute(out_axis_map);
    out_strides_ = out_.strides().Permute(out_axis_map);
}

// Squashes dimensions of reduction
//
// Example (in the case of a contiguous array):
// - in_shape_:             (5, 6, 2, 3, 4)
// - out_shape_:            (2, 3, 4)
// - in_squashed_shape:     (720)
// - out_squashed_shape:    (24)
void ReductionArg::Squash() {
    if (CHAINERX_DEBUG) {
        CHAINERX_ASSERT(in_shape_.ndim() == in_strides_.ndim());
        CHAINERX_ASSERT(out_shape_.ndim() == out_strides_.ndim());

        for (int8_t i = -1; i >= -out_shape_.ndim(); --i) {
            CHAINERX_ASSERT(in_shape_[in_shape_.ndim() + i] == out_shape_[out_shape_.ndim() + i]);
        }
    }

    // Squash out
    std::tuple<Shape, Axes> out_squashed_result = SquashShape(out_shape_, out_strides_);
    const Shape& out_squashed_shape = std::get<0>(out_squashed_result);
    const Axes& out_keep_axes = std::get<1>(out_squashed_result);
    Strides out_squashed_strides = GetSquashedStrides(out_strides_, out_keep_axes);

    // Squash in
    std::tuple<Shape, Axes> in_squashed_result = SquashShape(in_shape_, in_strides_);
    const Shape& in_squashed_shape = std::get<0>(in_squashed_result);
    const Axes& in_keep_axes = std::get<1>(in_squashed_result);
    Strides in_squashed_strides = GetSquashedStrides(in_strides_, in_keep_axes);

    CHAINERX_ASSERT(in_squashed_shape.ndim() == in_squashed_strides.ndim());
    CHAINERX_ASSERT(out_squashed_shape.ndim() == out_squashed_strides.ndim());

    in_strides_ = in_squashed_strides;
    out_strides_ = out_squashed_strides;
    in_shape_ = in_squashed_shape;
    out_shape_ = out_squashed_shape;
}

}  // namespace chainerx
