#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace elementwise_detail {

// Returns true if dimension i can be squashed for all strides, false otherwise.
template <typename... PackedStrides>
inline bool IsSquashableDimension(size_t i, const Shape& shape, const PackedStrides&... strides) {
    // If strides[i] * shape[i] != strides[i - 1] for any i for any strides, return false.
    // std::max seems to be faster than variadic function recursions.
    return !static_cast<bool>(std::max({(strides[i] * shape[i] != strides[i - 1])...}));
}

}  // namespace elementwise_detail

// Returns a subset of strides with elements corresponding to given axes.
// It can be used in conjunction with SquashedShape to obtain the squashed strides.
inline Strides SquashedStrides(const Strides& strides, const Axes& keep) {
    Strides squashed{};
    std::transform(keep.begin(), keep.end(), std::back_inserter(squashed), [&strides](int8_t axis) { return strides[axis]; });
    return squashed;
}

// Given arrays with equal shapes, returns a pair of a squashed shape with possibly fewer number of dimensions (but with equal total size)
// and axes that were kept in the procedure. Dimensions must be either successively contiguous or unit-length in order to be squashed as in
// the following examples.
//
// Example 1:
// Given arrays with Shape{2, 3}, all contiguous => Shape{6} and Axes{1}.
//
// Example 2:
// Given arrays with Shape{3, 2, 1, 2}, padded first dimension => Shape{3, 4} and Axes{0, 3}.
//
// Strided indexing spanning over multiple dimensions can be slow and may thus be preceded with this squash.
// Axes are needed to extract the subset of strides corresponding to the correct axes.
template <typename... Arrays>
std::pair<Shape, Axes> SquashedShape(const Array& array, Arrays&&... arrays) {
    Shape squashed{};
    Axes keep{};

    const Shape& shape = array.shape();
    switch (int8_t ndim = shape.ndim()) {
        case 0:
            squashed = shape;
            break;
        case 1:
            squashed = shape;
            keep.emplace_back(0);
            break;
        default:
            // Create a temporary shape with equal number of dimensions as in the original shape, but that will hold 1s where axes later can
            // be squashed.
            Shape compressed = shape;
            for (int8_t i = 1; i < ndim; ++i) {
                if (compressed[i - 1] == 1) {
                    continue;
                } else if (elementwise_detail::IsSquashableDimension(i, compressed, array.strides(), arrays.strides()...)) {
                    compressed[i] *= compressed[i - 1];
                    compressed[i - 1] = 1;
                    continue;
                }
                keep.emplace_back(i - 1);
            }
            if (compressed.back() != 1) {
                keep.emplace_back(ndim - 1);
            }

            if (keep.ndim() == ndim) {
                // No axes could be squashed.
                squashed = compressed;
                break;
            }
            // Squash compressed axes.
            std::copy_if(compressed.begin(), compressed.end(), std::back_inserter(squashed), [](int64_t dim) { return dim != 1; });
            break;
    }
    assert(squashed.ndim() == keep.ndim());
    return std::make_pair(squashed, keep);
}

}  // namespace xchainer
