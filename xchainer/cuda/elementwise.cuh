#pragma once

#include <cstdint>
#include <utility>

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
struct KernelLauncher {
    template <typename Kernel, typename ElementwiseImpl>
    __host__ void operator()(Kernel&& kernel, ElementwiseImpl&& impl, int64_t grid_size, int64_t block_size) {
        UnpackAndLaunch(
                std::forward<Kernel>(kernel), std::forward<ElementwiseImpl>(impl), grid_size, block_size, std::index_sequence_for<Ts...>());
    }

    template <typename Kernel, typename ElementwiseImpl, std::size_t... Is>
    __host__ void UnpackAndLaunch(
            Kernel&& kernel, ElementwiseImpl&& impl, int64_t grid_size, int64_t block_size, std::index_sequence<Is...>) {
        kernel<<<grid_size, block_size>>>(impl, arg.indexer, std::get<Is>(arg.iarrays)...);
    }

    ElementwiseKernelArg<Ts...>& arg;
};

}  // namespace elementwise_detail

template <typename ElementwiseImpl, typename... Ts>
void Elementwise(ElementwiseKernelArg<Ts...> arg, ElementwiseImpl&& impl) {
    auto kernel = &elementwise_detail::ElementwiseKernel<ElementwiseImpl, Ts...>;

    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(kernel).block_size;

    Indexer indexer = arg.indexer;

    int64_t total_size = indexer.total_size();
    int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    elementwise_detail::KernelLauncher<Ts...>{arg}(kernel, impl, grid_size, block_size);
}

}  // namespace cuda
}  // namespace xchainer
