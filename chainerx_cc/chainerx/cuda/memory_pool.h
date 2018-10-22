#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {

constexpr size_t kAllocationUnitSize = 512;

using AllocFunc = void (*)(void**, size_t);
using FreeFunc = void (*)(void*);

// Memory pool base.
// This class is thread safe.
template <AllocFunc Allocate, FreeFunc FreeMemory>
class MemoryPoolBase {
public:
    explicit MemoryPoolBase(int device_index) : device_index_{device_index} {}

    ~MemoryPoolBase() {
        for (const std::vector<void*>& free_list : free_bins_) {
            for (void* ptr : free_list) {
                FreeMemory(ptr);
            }
        }
        // Ideally, in_use_ should be empty, but it could happen that shared ptrs to memories allocated
        // by this memory pool are released after this memory pool is destructed.
        // Our approach is that we anyway free CUDA memories held by this memory pool here in such case.
        // Operators of arrays holding such memories will be broken, but are not supported.
        for (const auto& item : in_use_) {
            void* ptr = item.first;
            FreeMemory(ptr);
        }
    }

    void* Malloc(size_t bytesize) {
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
            // TODO(niboshi): Do device management with RAII
            int old_device{};
            CheckCudaError(cudaGetDevice(&old_device));
            CheckCudaError(cudaSetDevice(device_index_));

            Allocate(&ptr, allocation_size);

            CheckCudaError(cudaSetDevice(old_device));
        }

        {
            std::lock_guard<std::mutex> lock{in_use_mutex_};
            in_use_.emplace(ptr, index);
        }
        return ptr;
    }

    void Free(void* ptr) {
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

private:
    std::unordered_map<void*, size_t> in_use_;
    std::vector<std::vector<void*>> free_bins_;
    int device_index_;
    std::mutex in_use_mutex_;
    std::mutex free_bins_mutex_;
};

inline void MallocManaged(void** ptr, size_t bytesize) { CheckCudaError(cudaMallocManaged(ptr, bytesize, cudaMemAttachGlobal)); }
inline void HostAlloc(void** ptr, size_t bytesize) { CheckCudaError(cudaHostAlloc(ptr, bytesize, cudaHostAllocWriteCombined)); }

inline void Free(void* ptr) { CheckCudaError(cudaFree(ptr)); }
inline void FreeHost(void* ptr) { CheckCudaError(cudaFreeHost(ptr)); }

using MemoryPool = MemoryPoolBase<MallocManaged, Free>;
using PinnedMemoryPool = MemoryPoolBase<HostAlloc, FreeHost>;

}  // namespace cuda
}  // namespace chainerx
