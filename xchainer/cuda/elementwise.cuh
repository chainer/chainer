#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {
namespace cuda {
namespace elementwise_detail {

template <typename ElementwiseImpl, typename... IndexableArrays>
__global__ void ElementwiseKernel(ElementwiseImpl impl, Indexer indexer, IndexableArrays... args) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        impl.Operation(args[indexer]...);
    }
}

}  // namespace elementwise_detail

template <typename T, typename ElementwiseImpl, typename... Arrays>
void Elementwise(ElementwiseImpl&& impl, const Array& first, Arrays&&... rest) {
    // TODO(hvy): How to decide this?
    static int kMaxBlockSize = 512;

    Indexer indexer{first.shape()};

    int64_t total_size = indexer.total_size();
    int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    elementwise_detail::ElementwiseKernel<<<grid_size, block_size>>>(impl, indexer, IndexableArray<T>{first}, IndexableArray<T>{rest}...);
};

}  // namespace cuda
}  // namespace xchainer
