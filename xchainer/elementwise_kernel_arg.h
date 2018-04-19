#pragma once

#include <tuple>
#include <utility>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {

// Holds any number of arrays in a tuple for elementwise operations and an indexer matching their shapes.
// The tuples may be unpacked to match templetized kernels. See xchainer/native/elementwise.h.
template <typename... Ts>
struct ElementwiseKernelArg {
    explicit ElementwiseKernelArg(const Indexer& indexer, IndexableArray<Ts>&&... iarrays)
        : indexer{indexer}, iarrays{std::make_tuple(iarrays...)} {}

    Indexer indexer;
    std::tuple<IndexableArray<Ts>...> iarrays;

    static_assert(sizeof...(Ts) > 0, "Cannot create an elementwise kernel argument without any arrays.");
};

template <typename T, typename... Ts, typename... Arrays>
ElementwiseKernelArg<T, Ts...> MakeElementwiseKernelArg(const Array& first, Arrays&&... rest) {
    return ElementwiseKernelArg<T, Ts...>{Indexer{first.shape()}, IndexableArray<T>{first}, IndexableArray<Ts>{rest}...};
}

}  // namespace xchainer
