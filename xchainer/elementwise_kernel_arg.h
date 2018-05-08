/*
#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace internal {

// Returns true if dimension i can be compressed for all strides.
template <typename... PackedStrides>
inline bool IsCompressableDimension(size_t i, const Shape& shape, const PackedStrides&... strides) {
    // If strides[i] * shape[i] != strides[i - 1] for any i for any strides, return false.
    // std::max seems to be faster than variadic function recursions.
    return !static_cast<bool>(std::max({(strides[i] * shape[i] != strides[i - 1])...}));
}

// Returns a new set of strides where compressed dimensions are removed.
inline Strides ReducedStrides(const Shape& shape, const Strides& strides) {
    Strides reduced{};
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != 1) {
            // Dimension cannot be reduced, keep.
            reduced.push_back(strides[i]);
        }
    }
    return reduced;
}

}  // namespace internal

// Holds any number of arrays in a tuple for elementwise operations and an indexer matching their shapes.
// The tuples may be unpacked to match templetized kernels. See xchainer/native/elementwise.h.
template <typename... Ts>
struct ElementwiseKernelArg {
    explicit ElementwiseKernelArg(const Indexer<>& indexer, IndexableArray<Ts>&&... iarrays)
        : indexer{indexer}, iarrays{std::make_tuple(iarrays...)} {}

    Indexer<> indexer;
    std::tuple<IndexableArray<Ts>...> iarrays;

    static_assert(sizeof...(Ts) > 0, "Cannot create an elementwise kernel argument without any arrays.");
};

// Returns elementwise kernel-ready arguments for arrays.
// Arrays may consist of both inputs and outputs.
//
// Arrays are preprocessed so that kernel performance can be proved.
// E.g. compressible dimensions are reduced since indexing with strides is expensive.
template <typename T, typename... Ts, typename... Arrays>
ElementwiseKernelArg<T, Ts...> MakeElementwiseKernelArg(const Array& first, Arrays&&... rest) {
    const Shape& shape = first.shape();
    const Strides& strides = first.strides();

    int8_t ndim = shape.ndim();
    if (ndim <= 1) {
        return ElementwiseKernelArg<T, Ts...>{Indexer<>{shape}, IndexableArray<T>{first}, IndexableArray<Ts>{rest}...};
    }

    int8_t axis = -1;
    int8_t keepdims = 0;

    Shape comp_shape = shape;
    for (int8_t i = 1; i < ndim; ++i) {
        if (comp_shape[i - 1] == 1) {
            continue;
        }
        if (internal::IsCompressableDimension(i, comp_shape, strides, rest.strides()...)) {
            comp_shape[i] *= comp_shape[i - 1];
            comp_shape[i - 1] = 1;
            continue;
        }
        axis = i - 1;
        ++keepdims;
    }
    if (comp_shape.back() != 1) {
        axis = ndim - 1;
        ++keepdims;
    }

    if (keepdims == 1) {  // Compressed into a single dimensions.
        return ElementwiseKernelArg<T, Ts...>{Indexer<>{Shape{{comp_shape[axis]}}},
                                              IndexableArray<T>{first, Strides{{strides[axis]}}},
                                              IndexableArray<Ts>{rest, Strides{{rest.strides()[axis]}}}...};
    } else if (keepdims == ndim) {  // No dimensions compressed.
        return ElementwiseKernelArg<T, Ts...>{Indexer<>{shape}, IndexableArray<T>{first}, IndexableArray<Ts>{rest}...};
    }
    // Compressed some, but not not all dimensions.
    Shape reduced_shape{};
    std::copy_if(comp_shape.begin(), comp_shape.end(), std::back_inserter(reduced_shape), [](int64_t dim) { return dim != 1; });
    return ElementwiseKernelArg<T, Ts...>{Indexer<>{reduced_shape},
                                          IndexableArray<T>{first, internal::ReducedStrides(comp_shape, strides)},
                                          IndexableArray<Ts>{rest, internal::ReducedStrides(comp_shape, rest.strides())}...};
}

}  // namespace xchainer
*/
