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
void ElementwiseKernel(ElementwiseImpl&& impl, Indexer indexer, IndexableArray<Ts>... iarrays) {
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        impl(i, iarrays[indexer]...);
    }
}

// A callable struct that launches a kernel given its definition, loop body and the sizeof...(Ts) argument arrays that should be passed to
// it. When called, it first unpacks the argument arrays from a tuple to a parameter pack in order to pass them to the kernel.
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
    elementwise_detail::KernelLauncher<Ts...>{arg}(
            &elementwise_detail::ElementwiseKernel<ElementwiseImpl, Ts...>, std::forward<ElementwiseImpl>(impl));
}

}  // namespace native
}  // namespace xchainer
