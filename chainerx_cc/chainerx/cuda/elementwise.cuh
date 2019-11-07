#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <utility>

#include "chainerx/constant.h"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/index_iterator.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/shape.h"
#include "chainerx/squash_dims.h"

namespace chainerx {
namespace cuda {
namespace elementwise_detail {

template <int8_t Ndim, typename Op, typename... Ts>
__global__ void ElementwiseKernel(Op op, Indexer<Ndim> indexer, IndexableArray<Ts, Ndim>... args) {
    int64_t id = static_cast<int64_t>(blockIdx.x);
    int64_t size = static_cast<int64_t>(gridDim.x);
    int64_t block_dim = static_cast<int64_t>(blockDim.x);
    id = id * block_dim + static_cast<int64_t>(threadIdx.x);
    size *= block_dim;
    for (auto it = indexer.It(id, size); it; ++it) {
        op(it.raw_index(), cuda_internal::StorageToDataType<Ts>(args[it])...);
    }
}

template <int8_t Ndim, typename Op, typename... Ts, typename... Arrays>
void LaunchElementwiseKernel(Op&& op, const Shape& shape, const Axes& keep, const Arrays&... args) {
    // TODO(niboshi): Calculate kMaxBlockSize per device
    std::lock_guard<std::mutex> lock{*cuda_internal::g_mutex};

    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&ElementwiseKernel<Ndim, Op, Ts...>).block_size;

    int64_t total_size = shape.GetTotalSize();
    int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    ElementwiseKernel<Ndim, Op, Ts...><<<grid_size, block_size>>>(
            op, Indexer<Ndim>{shape}, IndexableArray<Ts, Ndim>{args, GetSquashedStrides(args.strides(), keep)}...);
}

}  // namespace elementwise_detail

template <typename... Ts, typename... Arrays, typename Op>
void Elementwise(Op&& op, const Arrays&... args) {
    static_assert(sizeof...(Ts) == sizeof...(Arrays), "Data types must be specified per Array. ");

    std::tuple<Shape, Axes> squashed_result = SquashShape(args...);
    const Shape& squashed = std::get<0>(squashed_result);
    const Axes& keep = std::get<1>(squashed_result);

#ifdef NDEBUG  // Optimize only in Release build to save time on development
    // TODO(hvy): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    switch (squashed.ndim()) {
        case 1:
            elementwise_detail::LaunchElementwiseKernel<1, Op, Ts...>(std::forward<Op>(op), squashed, keep, args...);
            return;
        case 2:
            elementwise_detail::LaunchElementwiseKernel<2, Op, Ts...>(std::forward<Op>(op), squashed, keep, args...);
            return;
        case 3:
            elementwise_detail::LaunchElementwiseKernel<3, Op, Ts...>(std::forward<Op>(op), squashed, keep, args...);
            return;
        case 4:
            elementwise_detail::LaunchElementwiseKernel<4, Op, Ts...>(std::forward<Op>(op), squashed, keep, args...);
            return;
    }
#endif  // NDEBUG

    elementwise_detail::LaunchElementwiseKernel<kDynamicNdim, Op, Ts...>(std::forward<Op>(op), squashed, keep, args...);
}

}  // namespace cuda
}  // namespace chainerx
