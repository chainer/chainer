#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

namespace chainerx {
namespace cuda {

constexpr size_t kAllocationUnitSize = 512;

// Memory pool.
// This class is thread safe.
class MemoryPool {
public:
    explicit MemoryPool(int device_index) : device_index_{device_index} {}
    ~MemoryPool();

    void* Malloc(size_t bytesize);
    void Free(void* ptr);

private:
    std::unordered_map<void*, size_t> in_use_;
    std::vector<std::vector<void*>> free_bins_;
    int device_index_;
    std::mutex in_use_mutex_;
    std::mutex free_bins_mutex_;
};

}  // namespace cuda
}  // namespace chainerx
