#pragma once

#include <memory>

namespace xchainer {

bool IsPointerCudaMemory(const void* ptr);
std::shared_ptr<void> Allocate(size_t size);
void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t size);
std::shared_ptr<void> MemoryFromBuffer(const std::shared_ptr<void>& src_ptr, size_t size);

}  // namespace
