#pragma once

#include <cstdint>
#include <tuple>

#include "xchainer/constant.h"
#include "xchainer/elementwise.h"
#include "xchainer/index_iterator.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace native {
namespace elementwise_detail {

template <int8_t Ndim, typename Op, typename... Ts>
void ElementwiseKernel(Op op, const Indexer<Ndim>& indexer, const IndexableArray<Ts, Ndim>&... args) {
    for (auto it = indexer.It(0, 1); it; ++it) {
        op(it.raw_index(), args[it]...);
    }
}

}  // namespace elementwise_detail

template <typename... Ts, typename... Arrays, typename Op>
void Elementwise(Op&& op, const Arrays&... args) {
    static_assert(sizeof...(Ts) == sizeof...(Arrays), "Data types must be specified per Array. ");

    Shape squashed{};
    Axes keep{};
    std::tie(squashed, keep) = SquashedShape(args...);

    // TODO(hvy): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    switch (squashed.ndim()) {
        case 1:
            elementwise_detail::ElementwiseKernel<1, Op, Ts...>(
                    op, Indexer<1>{squashed}, IndexableArray<Ts, 1>{args, SquashedStrides(args.strides(), keep)}...);
            break;
        case 2:
            elementwise_detail::ElementwiseKernel<2, Op, Ts...>(
                    op, Indexer<2>{squashed}, IndexableArray<Ts, 2>{args, SquashedStrides(args.strides(), keep)}...);
            break;
        case 3:
            elementwise_detail::ElementwiseKernel<3, Op, Ts...>(
                    op, Indexer<3>{squashed}, IndexableArray<Ts, 3>{args, SquashedStrides(args.strides(), keep)}...);
            break;
        case 4:
            elementwise_detail::ElementwiseKernel<4, Op, Ts...>(
                    op, Indexer<4>{squashed}, IndexableArray<Ts, 4>{args, SquashedStrides(args.strides(), keep)}...);
            break;
        default:
            elementwise_detail::ElementwiseKernel<kDynamicNdim, Op, Ts...>(
                    op, Indexer<kDynamicNdim>{squashed}, IndexableArray<Ts, kDynamicNdim>{args, SquashedStrides(args.strides(), keep)}...);
            break;
    }
}

}  // namespace native
}  // namespace xchainer
