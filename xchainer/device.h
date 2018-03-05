#pragma once

#include <memory>
#include <string>
#include <tuple>

#include "xchainer/backend.h"
#include "xchainer/scalar.h"

namespace xchainer {

class Array;

// Device base class.
// Note that these member functions may be called from the framework or user code.
class Device {
public:
    Device(Backend& backend, int index) : backend_(backend), index_(index) {}
    virtual ~Device() = default;

    // Allocates a memory chunk on this device.
    virtual std::shared_ptr<void> Allocate(size_t bytesize) = 0;

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
    // It returns a tuple of (data, offset).
    virtual std::tuple<std::shared_ptr<void>, size_t> TransferDataFrom(Device& src_device, const std::shared_ptr<void>& src_ptr,
                                                                       size_t offset, size_t bytesize) = 0;

    // Transfers the data from this device to the specified device.
    // It is usually preceded by a call to Backend::SupportsTransfer(), thus this function can assume transfer between the devices are
    // supported.
    //
    // It returns a tuple of (data, offset).
    virtual std::tuple<std::shared_ptr<void>, size_t> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr,
                                                                     size_t offset, size_t bytesize) = 0;

    // Creates a data buffer filled with the specified data on this device.
    //
    // It may allocate a new memory or return an alias.
    // src_ptr must reside in the main RAM.
    virtual std::shared_ptr<void> FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) = 0;

    virtual void Fill(Array& out, Scalar value) = 0;

    virtual void Add(const Array& lhs, const Array& rhs, Array& out) = 0;
    virtual void Mul(const Array& lhs, const Array& rhs, Array& out) = 0;

    virtual void Synchronize() = 0;

    // TODO(sonots): optimize string concat
    std::string name() const { return backend_.GetName() + ":" + std::to_string(index_); }

    Backend& backend() const { return backend_; }
    Context& context() const { return backend_.context(); }
    int index() const { return index_; }

protected:
    // Throws an exception if array devices are incompatible, else does nothing.
    template <typename... Arrays>
    void CheckDevicesCompatible(const Array& first, const Arrays&... rest) {
        CheckDevicesCompatible(first);
        CheckDevicesCompatible(rest...);
    }

private:
    void CheckDevicesCompatible(const Array& array);

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

// Scope object that switches the default device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_(internal::GetDefaultDeviceNoExcept()), exited_(false) {}
    explicit DeviceScope(Device& device) : DeviceScope() { SetDefaultDevice(&device); }

    // TODO(hvy): Maybe unnecessary.
    explicit DeviceScope(Backend* backend, int index = 0) : DeviceScope(backend->GetDevice(index)) {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    DeviceScope(DeviceScope&& other) : orig_(other.orig_), exited_(other.exited_) { other.exited_ = true; }

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultDevice(orig_);
            exited_ = true;
        }
    }

private:
    Device* orig_;
    bool exited_;
};

}  // namespace xchainer
