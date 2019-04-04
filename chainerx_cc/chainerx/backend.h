#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "chainerx/op.h"
#include "chainerx/op_registry.h"

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
    OpRegistry& op_registry() { return op_registry_; }

    // Returns the device for the given index.
    //
    // Throws out_of_range exception if index >= GetDeviceCount().
    Device& GetDevice(int index);

    // Queries if the backend supports data transfer between two devices.
    virtual bool SupportsTransfer(Device& src_device, Device& dst_device) = 0;

    // Calls the op implementation.
    template <typename OpType, typename... Args>
    auto CallOp(Args&&... args) {
        Op& op = op_registry_.GetOp<OpType>();
        return dynamic_cast<OpType&>(op).Call(std::forward<Args>(args)...);
    }

protected:
    // Returns a backend-specific global op registry.
    virtual OpRegistry& GetParentOpRegistry() = 0;

private:
    // Creates a new device.
    // This function is called from GetDevice().
    virtual std::unique_ptr<Device> CreateDevice(int index) = 0;

    Context& context_;

    std::vector<std::unique_ptr<Device>> devices_;

    std::mutex devices_mutex_;

    OpRegistry op_registry_;
};

}  // namespace chainerx
