#pragma once

#include <utility>

#include "xchainer/constant.h"
#include "xchainer/index_iterator.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace native {
namespace elementwise_detail {

template <int8_t Ndim, typename Op, typename... Ts>
void ElementwiseKernel(Op op, Indexer<Ndim> indexer, IndexableArray<Ts, Ndim>... args) {
    for (auto it = indexer.It(0, 1); it; ++it) {
        op(it.raw_index(), args[it]...);
    }
}

template <int8_t Ndim, typename Op, typename... Ts, typename... Arrays>
void LaunchElementwiseKernel(Op&& op, const Shape& shape, const Arrays&... args) {
    ElementwiseKernel<Ndim, Op, Ts...>(std::forward<Op>(op), Indexer<Ndim>{shape}, IndexableArray<Ts, Ndim>{args}...);
}

}  // namespace elementwise_detail

template <typename... Ts, typename... Arrays, typename Op>
void Elementwise(Op&& op, const Array& arg, const Arrays&... args) {
    static_assert(sizeof...(Ts) == 1 + sizeof...(Arrays), "Data types must be specified per Array. ");  // Ts includes the first array.
    // TODO(hvy): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    switch (arg.ndim()) {
        case 1:
            elementwise_detail::LaunchElementwiseKernel<1, Op, Ts...>(std::forward<Op>(op), arg.shape(), arg, args...);
            break;
        case 2:
            elementwise_detail::LaunchElementwiseKernel<2, Op, Ts...>(std::forward<Op>(op), arg.shape(), arg, args...);
            break;
        case 3:
            elementwise_detail::LaunchElementwiseKernel<3, Op, Ts...>(std::forward<Op>(op), arg.shape(), arg, args...);
            break;
        case 4:
            elementwise_detail::LaunchElementwiseKernel<4, Op, Ts...>(std::forward<Op>(op), arg.shape(), arg, args...);
            break;
        default:
            elementwise_detail::LaunchElementwiseKernel<kDynamicNdim, Op, Ts...>(std::forward<Op>(op), arg.shape(), arg, args...);
    }
}

}  // namespace native
}  // namespace xchainer
