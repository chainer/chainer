#pragma once

#include <cstdint>
#include <tuple>

#include "xchainer/array.h"
#include "xchainer/elementwise_kernel_arg.h"
#include "xchainer/indexer.h"

namespace xchainer {
namespace native {
namespace elementwise_detail {

template <typename Kernel, typename... Ts>
struct TupleUnpackDispatcher {
    template <typename ElementwiseImpl>
    void operator()(ElementwiseImpl&& impl, Indexer indexer) {
        Dispatch(impl, indexer, tuple, std::index_sequence_for<Ts...>());
    }

    template <typename ElementwiseImpl, std::size_t... Is>
    void Dispatch(ElementwiseImpl&& impl, Indexer indexer, const std::tuple<IndexableArray<Ts>...>& tup, std::index_sequence<Is...>) {
        kernel(impl, indexer, std::get<Is>(tup)...);
    }

    Kernel kernel;
    std::tuple<IndexableArray<Ts>...> tuple;
};

template <typename ElementwiseImpl, typename... Ts>
void ElementwiseKernel(ElementwiseImpl&& impl, Indexer indexer, IndexableArray<Ts>... args) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = 0; i < total_size; ++i) {
        indexer.Set(i);
        impl(args[indexer]...);
    }
}

}  // namespace elementwise_detail

template <typename... Ts, typename ElementwiseImpl>
void Elementwise(ElementwiseImpl&& impl, ElementwiseKernelArg<Ts...> arg) {
    using Kernel = std::function<void(ElementwiseImpl, Indexer indexer, IndexableArray<Ts>...)>;
    Kernel kernel = &elementwise_detail::ElementwiseKernel<ElementwiseImpl, Ts...>;
    elementwise_detail::TupleUnpackDispatcher<Kernel, Ts...>{kernel, arg.iarrays}(impl, arg.indexer);
}

}  // namespace native
}  // namespace xchainer
