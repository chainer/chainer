#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "chainerx/kernel.h"
#include "chainerx/kernel_registry.h"

namespace chainerx {

class Context;
class Device;

// Backend base class.
class Backend {
public:
    explicit Backend(Context& context);
    virtual ~Backend();

    Backend(const Backend&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend& operator=(Backend&&) = delete;

    // Initializes the backend instantce.
    virtual void Initialize();

    // Returns the name of this backend. This name should be unique within the context.
    virtual std::string GetName() const = 0;

    // Returns the number of available devices.
    //
    // This count is usually configurable by backend specific ways.
    virtual int GetDeviceCount() const = 0;

    // Returns the context.
    Context& context() const { return context_; }

    // Returns the op registry.
    KernelRegistry& kernel_registry() { return kernel_registry_; }

    // Returns the device for the given index.
    //
    // Throws out_of_range exception if index >= GetDeviceCount().
    Device& GetDevice(int index);

    // Returns whether the backend is a native device.
    virtual bool IsNative() const { return false; }

    // Queries if the backend supports data transfer between two devices.
    virtual bool SupportsTransfer(Device& src_device, Device& dst_device) = 0;

    // Calls the kernel implementation.
    template <typename KernelType, typename... Args>
    auto CallKernel(Args&&... args) {
        Kernel& kernel = kernel_registry_.GetKernel<KernelType>();
        return dynamic_cast<KernelType&>(kernel).Call(std::forward<Args>(args)...);
    }

protected:
    // Returns a backend-specific global kernel registry.
    virtual KernelRegistry& GetParentKernelRegistry() = 0;

private:
    // Creates a new device.
    // This function is called from GetDevice().
    virtual std::unique_ptr<Device> CreateDevice(int index) = 0;

    Context& context_;

    std::vector<std::unique_ptr<Device>> devices_;

    std::mutex devices_mutex_;

    KernelRegistry kernel_registry_;
};

}  // namespace chainerx
