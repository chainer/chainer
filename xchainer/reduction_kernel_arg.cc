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
    Shape reduce_shape{};  // Reduction dimensions
    Axes out_axis_map{};  // Mapping from effective output indices to actual output indices
    Shape new_out_shape{};
    // (Here "effective output indices" means source indices minus reduction indices.)

    // Example (in the case of has_kept_dims=false):
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 14, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 1, 2)
    // - new_out_shape:    (12, 14, 16)
    // Example (in the case of has_kept_dims=true):
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 1, 14, 1, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 2, 4)
    // - new_out_shape:    (12, 14, 16)

    {
        size_t i_axis = 0;
        size_t i_out_axis = 0;
        for (int8_t i = 0; i < in_.shape().ndim(); ++i) {
            if (i_axis < axis.size() && i == axis[i_axis]) {
                // i is to be reduced
                int64_t in_dim = in_.shape()[i];
                if (in_dim != 1) {
                    reduce_shape.emplace_back(in_dim);
                }
                ++i_axis;
                if (has_kept_dims) {
                    ++i_out_axis;
                }
            } else {
                // i is not to be reduced
                int64_t out_dim = out_.shape()[i_out_axis];
                if (out_dim != 1) {
                    out_axis_map.emplace_back(static_cast<int8_t>(i_out_axis));
                    new_out_shape.emplace_back(out_dim);
                }
                ++i_out_axis;
            }
        }
        assert(i_out_axis == out_.shape().size());
        assert(i_axis == axis.size());
    }
    // Inequality because 1-dim axes are eliminated.
    assert(reduce_shape.size() <= axis.size());
    assert(out_axis_map.size() <= in_.shape().size() - axis.size());
    assert(out_axis_map.size() == new_out_shape.size());

    // Calculate source axis permutation
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - axis_permutes:    (0, 2, 4, 1, 3)
    // - new_in_shape:     (12, 14, 16, 13, 15)
    Axes axis_permutes{};
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
    for (int8_t i : axis) {
        if (in_.shape()[i] != 1) {
            axis_permutes.emplace_back(i);
        }
    }
    assert(axis_permutes.size() <= in_.shape().size());  // Inequality because 1-dim axes are eliminated.

    // Calculate new source shape
    Shape new_in_shape{};
    for (int8_t i : axis_permutes) {
        new_in_shape.emplace_back(in_.shape()[i]);
    }

    // 1-dim axes must be eliminated
    assert(std::find(new_in_shape.begin(), new_in_shape.end(), 1) == new_in_shape.end());
    assert(std::find(new_out_shape.begin(), new_out_shape.end(), 1) == new_out_shape.end());

    Strides new_in_strides = in_.strides().Permute(axis_permutes);
    Strides new_out_strides = out_.strides().Permute(out_axis_map);

    in_strides_ = std::move(new_in_strides);
    out_strides_ = std::move(new_out_strides);
    in_shape_ = std::move(new_in_shape);
    out_shape_ = std::move(new_out_shape);
    reduce_shape_ = std::move(reduce_shape);
}

// Squashes dimensions of reduction
//
// Example (in the case of a contiguous array):
// - in_shape:     (2, 3, 4, 5, 6)
// - out_shape:    (2, 3, 4)
// - reduce_shape: (5, 6)
// - in_squashed_shape: (24, 30)
// - out_squashed_shape: (24)
// - reduce_squashed_shape: (30)
//
// Following equality always consists:
// in_squashed_shape.ndim() == out_squashed_shape.ndim() + reduce_squashed_shape.ndim()
//
// TODO(sonots): To achieve best performance optimization, squash dimensions of input and output individually, that is,
// in_squashed_shape.ndim() != out_squashed_shape.ndim() + reduce_squashed_shape.ndim()
// To do it, we have to revise implementation of ReductionKernel.
void ReductionArg::Squash() {
#ifndef NDEBUG
    assert(in_shape_.ndim() == out_shape_.ndim() + reduce_shape_.ndim());
    assert(in_shape_.ndim() == in_strides_.ndim());
    assert(out_shape_.ndim() == out_strides_.ndim());

    for (int8_t i = 0; i < out_shape_.ndim(); ++i) {
        assert(in_shape_[i] == out_shape_[i]);
    }
    for (int8_t i = 0; i < reduce_shape_.ndim(); ++i) {
        assert(in_shape_[out_shape_.ndim() + i] == reduce_shape_[i]);
    }
#endif

    // Some cases we can not squash further
    if (in_shape_.ndim() == 0) {
        assert(out_shape_.ndim() == 0 && reduce_shape_.ndim() == 0);
        return
    } else if (in_shape_.ndim() == 1) {
        assert((out_shape_.ndim() == 1 && reduce_shape_.ndim() == 0) || (out_shape_.ndim() == 0 && reduce_shape_.ndim() == 1));
        return
    } else if (in_shape_.ndim() == 2 && out_shape_.ndim() == 1) {
        assert(reduce_shape_.ndim() == 1);
        return
    }

    // Squash out
    Strides in_out_strides_{};  // former out_shape_.ndim() parts of in_strides_
    for (int8_t i = 0; i < out_strides_.ndim(); ++i) {
        in_out_strides_.emplace_back(in_strides_[i]);
    }
    std::tuple<Shape, Axes> out_squashed_result = SquashShape(out_shape_, in_out_strides_, out_strides_);
    const Shape& out_squashed_shape = std::get<0>(out_squashed_result);
    const Axes& out_keep_axes = std::get<1>(out_squashed_result);
    Strides out_squashed_strides = GetSquashedStrides(out_strides_, out_keep_axes);

    // Squash reduce
    Strides reduce_strides{};
    for (int8_t i = out_strides_.ndim(); i < in_strides_.ndim(); ++i) {
        reduce_strides.emplace_back(in_strides_[i]);
    }
    std::tuple<Shape, Axes> reduce_squashed_result = SquashShape(reduce_shape_, reduce_strides);
    const Shape& reduce_squashed_shape = std::get<0>(reduce_squashed_result);
    const Axes& reduce_keep_axes = std::get<1>(reduce_squashed_result);
    Strides reduce_squashed_strides = GetSquashedStrides(reduce_strides, reduce_keep_axes);

    // Merge out and reduce into input
    Shape in_squashed_shape{out_squashed_shape};
    Strides in_squashed_strides = GetSquashedStrides(in_strides_, out_keep_axes);
    for (int8_t i = 0; i < reduce_squashed_shape.ndim(); ++i) {
        in_squashed_shape.emplace_back(reduce_squashed_shape[i]);
        in_squashed_strides.emplace_back(reduce_squashed_strides[i]);
    }

#ifndef NDEBUG
    assert(in_squashed_shape.ndim() == out_squashed_shape.ndim() + reduce_squashed_shape.ndim());
    assert(in_squashed_shape.ndim() == in_squashed_strides.ndim());
    assert(out_squashed_shape.ndim() == out_squashed_strides.ndim());

    for (int8_t i = 0; i < out_squashed_shape.ndim(); ++i) {
        assert(in_squashed_shape[i] == out_squashed_shape[i]);
    }
    for (int8_t i = 0; i < reduce_squashed_shape.ndim(); ++i) {
        assert(in_squashed_shape[out_squashed_shape.ndim() + i] == reduce_squashed_shape[i]);
    }
#endif

    in_strides_ = std::move(in_squashed_strides);
    out_strides_ = std::move(out_squashed_strides);
    in_shape_ = std::move(in_squashed_shape);
    out_shape_ = std::move(out_squashed_shape);
    reduce_shape_ = std::move(reduce_squashed_shape);
}

}  // namespace xchainer
