#include "xchainer/reduction_kernel_arg.h"

#include <cassert>
#include <cstdint>
#include <tuple>

#include "xchainer/array.h"
#include "xchainer/shape.h"
#include "xchainer/squash_dims.h"
#include "xchainer/strides.h"

namespace xchainer {

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
                    out_shape_.emplace_back(out_dim);
                }
                ++i_out_axis;
            }
        }
        assert(i_out_axis == out_.shape().size());
        assert(i_axis == axis.size());
    }
    // Inequality because 1-dim axes are eliminated.
    assert(out_axis_map.size() <= in_.shape().size() - axis.size());
    assert(out_axis_map.size() == out_shape_.size());

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
    assert(axis_permutes.size() <= in_.shape().size());  // Inequality because 1-dim axes are eliminated.

    // Calculate new source shape
    for (int8_t i : axis_permutes) {
        in_shape_.emplace_back(in_.shape()[i]);
    }

    // 1-dim axes must be eliminated
    assert(std::find(in_shape_.begin(), in_shape_.end(), 1) == in_shape_.end());
    assert(std::find(out_shape_.begin(), out_shape_.end(), 1) == out_shape_.end());

    in_strides_ = in_.strides().Permute(axis_permutes);
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
#ifndef NDEBUG
    assert(in_shape_.ndim() == in_strides_.ndim());
    assert(out_shape_.ndim() == out_strides_.ndim());

    for (int8_t i = -1; i >= -out_shape_.ndim(); --i) {
        assert(in_shape_[in_shape_.ndim() + i] == out_shape_[out_shape_.ndim() + i]);
    }
#endif

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

#ifndef NDEBUG
    assert(in_squashed_shape.ndim() == in_squashed_strides.ndim());
    assert(out_squashed_shape.ndim() == out_squashed_strides.ndim());
#endif

    in_strides_ = in_squashed_strides;
    out_strides_ = out_squashed_strides;
    in_shape_ = in_squashed_shape;
    out_shape_ = out_squashed_shape;
}

}  // namespace xchainer
