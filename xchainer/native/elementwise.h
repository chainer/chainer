#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/indexer.h"

namespace xchainer {
namespace native {
namespace elementwise_detail {

template <typename ElementwiseImpl, typename... IndexableArrays>
void ElementwiseKernel(ElementwiseImpl&& impl, Indexer indexer, IndexableArrays&&... args) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = 0; i < total_size; ++i) {
        indexer.Set(i);
        impl(args[indexer]...);
    }
}

}  // namespace elementwise_detail

template <typename T, typename ElementwiseImpl, typename... IndexableArrays>
void Elementwise(ElementwiseImpl&& impl, const Array& first, IndexableArrays&&... args) {
    elementwise_detail::ElementwiseKernel(impl, Indexer{first.shape()}, IndexableArray<T>(first), IndexableArray<T>{args}...);
}

}  // namespace native
}  // namespace xchainer
