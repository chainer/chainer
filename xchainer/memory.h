#pragma once

#include <memory>

#include "xchainer/device.h"

namespace xchainer {

bool IsPointerCudaMemory(const void* ptr);
std::shared_ptr<void> Allocate(const Device& device, size_t bytesize);
void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize);
std::shared_ptr<void> MemoryFromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize);

}  // namespace xchainer
