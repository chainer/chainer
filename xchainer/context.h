#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unoredered_map>

#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"

namespace xchainer {

class Context {
public:
    // Gets the backend specified by the name.
    // If the backend does not exist, this function automatically creates it.
    Backend& GetBackend(const std::string& backend_name);

    // Gets the device specified by the device ID.
    // If the backend and/or device do not exist, this function automatically creates them.
    Device& GetDevice(const DeviceId& device_id);

    // Gets/sets the default device of this context.
    void set_default_device(Device& device) {
        if (this != &device.backend().context()) {
            throw ContextError("Context mismatch.");
        }
        std::lock_guard<std::recursive_mutex> lock{mutex_};
        default_device_ = &device;
    }

    Device& default_device() const {
        std::lock_guard<std::recursive_mutex> lock{mutex_};
        if (default_device_ == nullptr) {
            throw ContextError("Global default device is not set.");
        }
        return *default_device_;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
    Device* default_device_ = nullptr;
    std::recursive_mutex mutex_;
};

// Gets/sets the context that used by default when current context is not set.
Context& GetGlobalDefaultContext();
void SetGlobalDefaultContext(Context* context);

// Gets/sets thread local context.
Context& GetDefaultContext();
void SetDefaultContext(Context* context);

}  // namespace xchainer
