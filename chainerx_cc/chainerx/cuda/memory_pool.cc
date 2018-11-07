#include "chainerx/cuda/memory_pool.h"

#include <mutex>
#include <unordered_map>
#include <vector>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {

MallocStatus DeviceMemoryAllocator::Malloc(void** ptr, size_t bytesize) {
    cudaError_t status = cudaMallocManaged(ptr, bytesize, cudaMemAttachGlobal);
    switch (status) {
        case cudaSuccess:
            return MallocStatus::kSuccess;
        case cudaErrorMemoryAllocation:
            return MallocStatus::kErrorMemoryAllocation;
        default:
            Throw(status);
    }
    CHAINERX_NEVER_REACH();
}

MallocStatus PinnedMemoryAllocator::Malloc(void** ptr, size_t bytesize) {
    cudaError_t status = cudaHostAlloc(ptr, bytesize, cudaHostAllocWriteCombined);
    switch (status) {
        case cudaSuccess:
            return MallocStatus::kSuccess;
        case cudaErrorMemoryAllocation:
            return MallocStatus::kErrorMemoryAllocation;
        default:
            Throw(status);
    }
    CHAINERX_NEVER_REACH();
}

MemoryPool::~MemoryPool() {
    // NOTE: CudaSetDeviceScope is not available at dtor because it may throw
    int orig_device_index{0};
    cudaGetDevice(&orig_device_index);
    cudaSetDevice(device_index_);

    for (const std::vector<void*>& free_list : free_bins_) {
        for (void* ptr : free_list) {
            allocator_->Free(ptr);
        }
    }
    // Ideally, in_use_ should be empty, but it could happen that shared ptrs to memories allocated
    // by this memory pool are released after this memory pool is destructed.
    // Our approach is that we anyway free CUDA memories held by this memory pool here in such case.
    // Operators of arrays holding such memories will be broken, but are not supported.
    for (const auto& item : in_use_) {
        void* ptr = item.first;
        allocator_->Free(ptr);
    }

    cudaSetDevice(orig_device_index);
}

void MemoryPool::FreeAllBlocks() {
    CudaSetDeviceScope scope{device_index_};
    std::lock_guard<std::mutex> lock{free_bins_mutex_};
    for (const std::vector<void*>& free_list : free_bins_) {
        for (void* ptr : free_list) {
            allocator_->Free(ptr);
        }
    }
    free_bins_.clear();
}

void* MemoryPool::Malloc(size_t bytesize) {
    if (bytesize == 0) {
        return nullptr;
    }

    size_t index = (bytesize - 1) / kAllocationUnitSize;

    void* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock{free_bins_mutex_};
        if (free_bins_.size() <= index) {
            free_bins_.resize(index + 1);
        }
        std::vector<void*>& free_list = free_bins_[index];

        if (!free_list.empty()) {
            ptr = free_list.back();
            CHAINERX_ASSERT(ptr != nullptr);
            free_list.pop_back();
        }
    }

    if (ptr == nullptr) {
        size_t allocation_size = (index + 1) * kAllocationUnitSize;
        CudaSetDeviceScope scope{device_index_};
        MallocStatus status = allocator_->Malloc(&ptr, allocation_size);
        if (status == MallocStatus::kErrorMemoryAllocation) {
            FreeAllBlocks();
            status = allocator_->Malloc(&ptr, allocation_size);
            if (status == MallocStatus::kErrorMemoryAllocation) {
                // TODO(sonots): Include total pooled bytes in the error message
                throw OutOfMemoryError{allocation_size};
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock{in_use_mutex_};
        in_use_.emplace(ptr, index);
    }
    return ptr;
}

void MemoryPool::Free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    size_t index{};

    {
        std::lock_guard<std::mutex> lock{in_use_mutex_};
        auto it = in_use_.find(ptr);
        if (it == in_use_.end()) {
            throw ChainerxError{"Cannot free out-of-pool memory"};
        }
        index = it->second;
        in_use_.erase(it);
    }

    {
        std::lock_guard<std::mutex> lock{free_bins_mutex_};
        std::vector<void*>& free_list = free_bins_.at(index);
        free_list.push_back(ptr);
    }
}

}  // namespace cuda
}  // namespace chainerx
