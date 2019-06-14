#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {
namespace squash_dims_detail {

// Returns true if dimension i can be squashed for all strides, false otherwise.
template <typename... PackedStrides>
inline bool IsSquashableDimension(size_t i, const Shape& shape, const PackedStrides&... strides) {
    // If strides[i] * shape[i] != strides[i - 1] for any i for any strides, return false.
    // std::max seems to be faster than variadic function recursions.
    return !static_cast<bool>(std::max({(strides[i] * shape[i] != strides[i - 1])...}));
}

}  // namespace squash_dims_detail

// Returns a subset of strides with elements corresponding to the given axes that were kept after squashing a shape with SquashShape.
// It should therefore be called with the resulting keep axes from SquashShape and the strides from the same arrays.
inline Strides GetSquashedStrides(const Strides& strides, const Axes& keep) {
    Strides squashed{};
    std::transform(keep.begin(), keep.end(), std::back_inserter(squashed), [&strides](int8_t axis) { return strides[axis]; });
    return squashed;
}

// Given a common shape and respective strides of arrays, returns a tuple with a squashed shape with possibly fewer number of
// dimensions (but with equal total size) and axes that were kept in the procedure. Dimensions must be either successively contiguous or
// unit-length in order to be squashed as in the following examples.
//
// Example 1:
// Given Shape{2, 3}, all contiguous strides => Shape{6} and Axes{1}.
//
// Example 2:
// Given Shape{3, 2, 1, 2}, strides with padded first dimension => Shape{3, 4} and Axes{0, 3}.
//
// Strided indexing spanning over multiple dimensions can be slow and may thus be preceded with this squash.
// Axes are needed to extract the subset of strides corresponding to the correct axes using GetSquashedStrides.
template <typename... PackedStrides>
std::tuple<Shape, Axes> SquashShape(const Shape& shape, const PackedStrides&... strides) {
    Shape squashed{};
    Axes keep{};

    switch (int8_t ndim = shape.ndim()) {
        case 0:
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
                    // Do nothing.
                } else if (squash_dims_detail::IsSquashableDimension(i, compressed, strides...)) {
                    compressed[i] *= compressed[i - 1];
                    compressed[i - 1] = 1;
                } else {
                    keep.emplace_back(i - 1);
                }
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
    CHAINERX_ASSERT(squashed.ndim() == keep.ndim());
    return std::make_tuple(squashed, keep);
}

template <typename... Arrays>
std::tuple<Shape, Axes> SquashShape(const Array& array, Arrays&&... arrays) {
    return SquashShape(array.shape(), array.strides(), arrays.strides()...);
}

}  // namespace chainerx
