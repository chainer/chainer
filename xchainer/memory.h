#pragma once

#include <memory>

#include "xchainer/device_id.h"

namespace xchainer {
namespace internal {

bool IsPointerCudaMemory(const void* ptr);
std::shared_ptr<void> Allocate(const DeviceId& device_id, size_t bytesize);
void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize);
std::shared_ptr<void> MemoryFromBuffer(const DeviceId& device_id, const std::shared_ptr<void>& src_ptr, size_t bytesize);

}  // namespace internal
}  // namespace xchainer
