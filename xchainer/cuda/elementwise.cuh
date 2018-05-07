#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include "xchainer/constant.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/index_iterator.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace cuda {
namespace elementwise_detail {

template <int8_t Ndim, typename Op, typename... Ts>
__global__ void ElementwiseKernel(Op op, Indexer<Ndim> indexer, IndexableArray<Ts, Ndim>... args) {
    for (auto it = indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        op(it.raw_index(), args[it]...);
    }
}

template <int8_t Ndim, typename Op, typename... Ts, typename... Arrays>
void LaunchElementwiseKernel(Op&& op, const Shape& shape, const Arrays&... args) {
    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&ElementwiseKernel<Ndim, Op, Ts...>).block_size;

    int64_t total_size = shape.GetTotalSize();
    int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    ElementwiseKernel<Ndim, Op, Ts...><<<grid_size, block_size>>>(op, Indexer<Ndim>{shape}, IndexableArray<Ts, Ndim>{args}...);
}

}  // namespace elementwise_detail

template <typename... Ts, typename... Arrays, typename Op>
void Elementwise(Op&& op, const Array& arg, const Arrays&... args) {
    static_assert(sizeof...(Ts) == 1 + sizeof...(Arrays), "Data types must be specified per Array. ");  // Ts includes the first array.
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

}  // namespace cuda
}  // namespace xchainer
