#include "xchainer/reduction_kernel_arg.h"

#include <cassert>
#include <cstdint>
#include <tuple>

#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace internal {

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
// TODO(sonots): squash input and output dimensions individually to achieve best performance
SquashReductionArg SquashReductionShapeAndStrides(const SquashReductionArg& arg) {
#ifndef NDEBUG
    assert(arg.in_shape.ndim() == arg.out_shape.ndim() + arg.reduce_shape.ndim());
    assert(arg.in_shape.ndim() == arg.in_strides.ndim());
    assert(arg.out_shape.ndim() == arg.out_strides.ndim());

    for (int8_t i = 0; i < arg.out_shape.ndim(); ++i) {
        assert(arg.in_shape[i] == arg.out_shape[i]);
    }
    for (int8_t i = 0; i < arg.reduce_shape.ndim(); ++i) {
        assert(arg.in_shape[arg.out_shape.ndim() + i] == arg.reduce_shape[i]);
    }
#endif

    // Skip squashing for apparent cases:
    // if (in_shape.ndim() == 0) {
    //     assert(out_shape.ndim() == 0 && reduce_shape.ndim() == 0);
    //     return std::make_tuple(in_shape, in_strides, reduce_shape);
    // } else if (in_shape.ndim() == 1) {
    //     assert((out_shape.ndim() == 1 && reduce_shape.ndim() == 0) || (out.shape.ndim() == 0 && reduce_shape.ndim == 1));
    //     return std::make_tuple(in_shape, in_strides, reduce_shape);
    // } else if (in_shape.ndim() == 2 && out_shape.ndim() == 1) {
    //     assert(reduce_shape.ndim() == 1);
    //     return std::make_tuple(in_shape, in_strides, reduce_shape);
    // }

    // Squash out
    // Only out_shape.ndim() elements in in_strides are seen in SquashShape
    std::tuple<Shape, Axes> out_squashed_result = SquashShape(arg.out_shape, arg.in_strides, arg.out_strides);
    const Shape& out_squashed_shape = std::get<0>(out_squashed_result);
    const Axes& out_keep_axes = std::get<1>(out_squashed_result);
    Strides out_squashed_strides = GetSquashedStrides(arg.out_strides, out_keep_axes);

    // Squash reduce
    Strides reduce_strides{};
    for (int8_t i = arg.out_strides.ndim(); i < arg.in_strides.ndim(); ++i) {
        reduce_strides.emplace_back(arg.in_strides[i]);
    }
    std::tuple<Shape, Axes> reduce_squashed_result = SquashShape(arg.reduce_shape, reduce_strides);
    const Shape& reduce_squashed_shape = std::get<0>(reduce_squashed_result);
    const Axes& reduce_keep_axes = std::get<1>(reduce_squashed_result);
    Strides reduce_squashed_strides = GetSquashedStrides(reduce_strides, reduce_keep_axes);

    // Merge out and reduce into input
    Shape in_squashed_shape{out_squashed_shape};
    Strides in_squashed_strides = GetSquashedStrides(arg.in_strides, out_keep_axes);
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

    return SquashReductionArg{std::move(in_squashed_strides),
                              std::move(out_squashed_strides),
                              std::move(in_squashed_shape),
                              std::move(out_squashed_shape),
                              std::move(reduce_squashed_shape)};
}  // namespace internal

}  // namespace internal
}  // namespace xchainer
