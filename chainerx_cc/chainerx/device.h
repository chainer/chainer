#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <absl/types/optional.h>

#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {

class Array;

// Device base class.
// Note that these member functions may be called from the framework or user code.
class Device {
public:
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = delete;

    // Allocates a memory chunk on this device.
    virtual std::shared_ptr<void> Allocate(size_t bytesize) = 0;

    // Makes an array data pointer from a foreign pointer without copying.
    // May throw an error if the foreign pointer is invalid for this device.
    virtual std::shared_ptr<void> MakeDataFromForeignPointer(const std::shared_ptr<void>& data) { return data; }

    // Copies the data between devices.
    // The other device may or may not be the same as this device.
    // The caller must guarantee that:
    // - Data transfer between the devices is supported by this backend.
    //   That is, Backend::SupportsTransfer must return true for the devices.
    // - Memory regions are not overlapped.
    virtual void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) = 0;
    virtual void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) = 0;

    // Transfers the data from the specified device to this device.
    // It is usually preceded by a call to Backend::SupportsTransfer(), thus this function can assume transfer between the devices are
    // supported.
    //
    // It returns a pointer to the allocated memory.
    virtual std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) = 0;

    // Transfers the data from this device to the specified device.
    // It is usually preceded by a call to Backend::SupportsTransfer(), thus this function can assume transfer between the devices are
    // supported.
    //
    // It returns a pointer to the allocated memory.
    virtual std::shared_ptr<void> TransferDataTo(
            Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) = 0;

    // Creates a data buffer filled with the specified data on this device.
    //
    // It may allocate a new memory or return an alias.
    // src_ptr must reside in the host memory.
    virtual std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) = 0;

    virtual void Synchronize() = 0;

    // TODO(sonots): optimize string concat
    std::string name() const { return backend_.GetName() + ":" + std::to_string(index_); }

    Backend& backend() const { return backend_; }
    Context& context() const { return backend_.context(); }
    int index() const { return index_; }

    // Throws an exception if array devices are incompatible, else does nothing.
    template <typename... Arrays>
    void CheckDevicesCompatible(const Array& first, const Arrays&... rest) {
        CheckDevicesCompatible(first);
        CheckDevicesCompatible(rest...);
    }

    void CheckDevicesCompatible(const Array& array);

protected:
    Device(Backend& backend, int index) : backend_{backend}, index_{index} {}

private:
    Backend& backend_;
    int index_;
};

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept;

}  // namespace internal

// Gets the default device. If the default device is null in this thread, it sets and returns the "native:0" device of the default context.
Device& GetDefaultDevice();

// Sets thread local device.
//
// Raises ContextError if context mismatches between given device and default context.
void SetDefaultDevice(Device* device);

void CheckEqual(const Device& lhs, const Device& rhs);

// Scope object that switches the default device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_{internal::GetDefaultDeviceNoExcept()}, exited_{false} {}
    explicit DeviceScope(Device& device) : DeviceScope{} { SetDefaultDevice(&device); }

    // TODO(hvy): Maybe unnecessary.
    explicit DeviceScope(Backend* backend, int index = 0) : DeviceScope{backend->GetDevice(index)} {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    DeviceScope(DeviceScope&& other) noexcept : orig_{other.orig_}, exited_{other.exited_} { other.exited_ = true; }

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            try {
                SetDefaultDevice(orig_);
            } catch (...) {
                CHAINERX_NEVER_REACH();
            }
            exited_ = true;
        }
    }

private:
    Device* orig_;
    bool exited_;
};

}  // namespace chainerx
