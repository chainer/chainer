#include "chainerx/native/native_device.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "chainerx/device.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace native {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) {
    if (bytesize == 0) {
        return std::shared_ptr<void>{nullptr};
    }
    return std::shared_ptr<uint8_t>(new uint8_t[bytesize], std::default_delete<uint8_t[]>());
}

void NativeDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    CHAINERX_ASSERT(nullptr != dynamic_cast<NativeDevice*>(&src_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

void NativeDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    CHAINERX_ASSERT(nullptr != dynamic_cast<NativeDevice*>(&dst_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

std::shared_ptr<void> NativeDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> NativeDevice::TransferDataTo(
        Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    return dst_device.TransferDataFrom(*this, src_ptr, offset, bytesize);
}

std::shared_ptr<void> NativeDevice::FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)bytesize;  // unused
    return src_ptr;
}

}  // namespace native
}  // namespace chainerx
