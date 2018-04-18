#pragma once

#include <tuple>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {

template <typename... Ts>
struct ElementwiseKernelArg {
    explicit ElementwiseKernelArg(Indexer indexer, IndexableArray<Ts>&&... iarrays)
        : indexer{indexer}, iarrays{std::tuple<IndexableArray<Ts>...>{iarrays...}} {}

    Indexer indexer;
    std::tuple<IndexableArray<Ts>...> iarrays;

    static_assert(sizeof...(Ts) > 0, "Cannot create an elementwise kernel argument without any arrays.");
};

template <typename T>
ElementwiseKernelArg<T> MakeElementwiseKernelArg(const Array& first) {
    return ElementwiseKernelArg<T>{Indexer{first.shape()}, IndexableArray<T>{first}};
}

template <typename T, typename... Ts, typename... Arrays>
ElementwiseKernelArg<T, Ts...> MakeElementwiseKernelArg(const Array& first, Arrays&&... rest) {
    return ElementwiseKernelArg<T, Ts...>{Indexer{first.shape()}, IndexableArray<T>{first}, IndexableArray<Ts>{rest}...};
}

}  // namespace xchainer
