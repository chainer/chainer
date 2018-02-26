#pragma once

#include <memory>

#include "xchainer/device.h"

namespace xchainer {
namespace internal {

bool IsPointerCudaMemory(const void* ptr);
std::shared_ptr<void> Allocate(Device& device, size_t bytesize);
void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize);
std::shared_ptr<void> MemoryFromBuffer(Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize);

}  // namespace internal
}  // namespace xchainer
