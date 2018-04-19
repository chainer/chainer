#pragma once

#include <algorithm>
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
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size(); i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        impl(i, args[indexer]...);
    }
}

// Unpacks the argument arrays from a tuple to a parameter pack and launches a kernel. See xchainer/native/elementwise.h.
template <typename... Ts>
struct KernelLauncher {
    template <typename Kernel, typename ElementwiseImpl>
    __host__ void operator()(Kernel&& kernel, ElementwiseImpl&& impl) {
        UnpackAndLaunch(std::forward<Kernel>(kernel), std::forward<ElementwiseImpl>(impl), std::index_sequence_for<Ts...>());
    }

    template <typename Kernel, typename ElementwiseImpl, std::size_t... Is>
    __host__ void UnpackAndLaunch(Kernel&& kernel, ElementwiseImpl&& impl, std::index_sequence<Is...>) {
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(kernel).block_size;

        int64_t total_size = arg.indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        kernel<<<grid_size, block_size>>>(std::forward<ElementwiseImpl>(impl), arg.indexer, std::get<Is>(arg.iarrays)...);
    }

    ElementwiseKernelArg<Ts...>& arg;
};

}  // namespace elementwise_detail

template <typename ElementwiseImpl, typename... Ts>
void Elementwise(ElementwiseKernelArg<Ts...> arg, ElementwiseImpl&& impl) {
    elementwise_detail::KernelLauncher<Ts...>{arg}(
            &elementwise_detail::ElementwiseKernel<ElementwiseImpl, Ts...>, std::forward<ElementwiseImpl>(impl));
}

}  // namespace cuda
}  // namespace xchainer
