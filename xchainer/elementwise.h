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

// Returns true if dimension i can be compressed for all strides.
template <typename... PackedStrides>
inline bool IsCompressibleDimension(size_t i, const Shape& shape, const PackedStrides&... strides) {
    // If strides[i] * shape[i] != strides[i - 1] for any i for any strides, return false.
    // std::max seems to be faster than variadic function recursions.
    return !static_cast<bool>(std::max({(strides[i] * shape[i] != strides[i - 1])...}));
}

}  // namespace elementwise_detail

// Returns a subset of strides with elements corresponding to given axes.
// It is used in conjunction with ReducedShape to obtain the reduced strides.
inline Strides Reduce(const Strides& strides, const Axes& keep) {
    Strides reduced{};
    std::transform(keep.begin(), keep.end(), std::back_inserter(reduced), [&strides](int8_t axis) { return strides[axis]; });
    return reduced;
}

// Returns a reduced shape with indices of the axes that were kept.
template <typename... Arrays>
std::pair<Shape, Axes> ReducedShape(const Array& array, Arrays&&... arrays) {
    Shape reduced{};
    Axes keep{};

    const Shape& shape = array.shape();
    switch (int8_t ndim = shape.ndim()) {
        case 0:
            reduced = shape;
            break;
        case 1:
            reduced = shape;
            keep.emplace_back(0);
            break;
        default:
            Shape compressed = shape;
            for (int8_t i = 1; i < ndim; ++i) {
                if (compressed[i - 1] == 1) {
                    continue;
                } else if (elementwise_detail::IsCompressibleDimension(i, compressed, array.strides(), arrays.strides()...)) {
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
                // No dimensions could be reduced.
                reduced = compressed;
                break;
            }
            // Reduce compressed dimensions.
            std::copy_if(compressed.begin(), compressed.end(), std::back_inserter(reduced), [](int64_t dim) { return dim != 1; });
            break;
    }
    assert(reduced.ndim() == keep.ndim());
    return std::make_pair(reduced, keep);
}

}  // namespace xchainer
