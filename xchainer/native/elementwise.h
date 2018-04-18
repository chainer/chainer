#pragma once

#include <cstdint>
#include <utility>

#include "xchainer/elementwise_kernel_arg.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {
namespace native {
namespace elementwise_detail {

template <typename ElementwiseImpl, typename... Ts>
void ElementwiseKernel(ElementwiseImpl&& loop_body, Indexer indexer, IndexableArray<Ts>... iarrays) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = 0; i < total_size; ++i) {
        indexer.Set(i);
        loop_body(iarrays[indexer]...);
    }
}

template <typename... Ts>
struct KernelLauncher {
    template <typename Kernel, typename ElementwiseImpl>
    void operator()(Kernel&& kernel, ElementwiseImpl&& impl) {
        UnpackAndLaunch(std::forward<Kernel>(kernel), std::forward<ElementwiseImpl>(impl), std::index_sequence_for<Ts...>());
    }

    template <typename Kernel, typename ElementwiseImpl, std::size_t... Is>
    void UnpackAndLaunch(Kernel&& kernel, ElementwiseImpl&& impl, std::index_sequence<Is...>) {
        kernel(std::forward<ElementwiseImpl>(impl), arg.indexer, std::get<Is>(arg.iarrays)...);
    }

    ElementwiseKernelArg<Ts...>& arg;
};

}  // namespace elementwise_detail

template <typename ElementwiseImpl, typename... Ts>
void Elementwise(ElementwiseKernelArg<Ts...> arg, ElementwiseImpl&& impl) {
    using Kernel = std::function<void(ElementwiseImpl, Indexer indexer, IndexableArray<Ts>...)>;
    Kernel kernel = &elementwise_detail::ElementwiseKernel<ElementwiseImpl, Ts...>;
    elementwise_detail::KernelLauncher<Ts...>{arg}(kernel, impl);
}

}  // namespace native
}  // namespace xchainer
