#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/elementwise_kernel_arg.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {
namespace cuda {
namespace elementwise_detail {

template <typename ElementwiseImpl, typename... Ts>
__global__ void ElementwiseKernel(ElementwiseImpl impl, Indexer indexer, IndexableArray<Ts>... args) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        impl(args[indexer]...);
    }
}

template <typename... Ts>
struct TupleUnpackDispatcher {
    template <typename Kernel, typename ElementwiseImpl>
    __host__ void operator()(Kernel kernel, ElementwiseImpl&& impl, Indexer indexer, int64_t grid_size, int64_t block_size) {
        Dispatch(kernel, impl, indexer, grid_size, block_size, tuple, std::index_sequence_for<Ts...>());
    }

    template <typename Kernel, typename ElementwiseImpl, std::size_t... Is>
    __host__ void Dispatch(
            Kernel kernel,
            ElementwiseImpl&& impl,
            Indexer indexer,
            int64_t grid_size,
            int64_t block_size,
            const std::tuple<IndexableArray<Ts>...>& tup,
            std::index_sequence<Is...>) {
        kernel<<<grid_size, block_size>>>(impl, indexer, std::get<Is>(tup)...);
    }

    std::tuple<IndexableArray<Ts>...> tuple;
};

}  // namespace elementwise_detail

template <typename... Ts, typename ElementwiseImpl>
void Elementwise(ElementwiseImpl&& impl, ElementwiseKernelArg<Ts...> arg) {
    auto kernel = &elementwise_detail::ElementwiseKernel<ElementwiseImpl, Ts...>;

    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(kernel).block_size;

    Indexer indexer = arg.indexer;

    int64_t total_size = indexer.total_size();
    int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    elementwise_detail::TupleUnpackDispatcher<Ts...>{arg.iarrays}(kernel, impl, indexer, grid_size, block_size);
}

}  // namespace cuda
}  // namespace xchainer
