#include "xchainer/cuda/memory_pool.h"

#include <vector>
#include <unordered_map>

#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

void* MemoryPool::Malloc(size_t bytesize) {
    if (bytesize == 0) {
        return nullptr;
    }

    size_t index = (bytesize - 1) / kAllocationUnitSize;

    if (free_bins_.size() <= index) {
        free_bins_.resize(index + 1);
    }
    std::vector<void*>& free_list = free_bins_[index];

    void* ptr = nullptr;
    if (!free_list.empty()) {
        ptr = free_list.back();
        free_list.pop_back();
    } else {
        size_t allocation_size = (index + 1) * kAllocationUnitSize;
        CheckCudaError(cudaSetDevice(device_index_));
        CheckCudaError(cudaMallocManaged(&ptr, allocation_size, cudaMemAttachGlobal));
    }
    in_use_.emplace(ptr, index);
    return ptr;
}

void MemoryPool::Free(void *ptr) {
    auto it = in_use_.find(ptr);
    if (it == in_use_.end()) {
        throw XchainerError{"Cannot free out-of-pool memory"};
    }

    size_t index = it->second;
    std::vector<void*>& free_list = free_bins_.at(index);
    free_list.push_back(ptr);

    in_use_.erase(it);
}

}  // namespace cuda
}  // namespace xchainer
