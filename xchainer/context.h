#pragma once

#include <string>

#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"

namespace xchainer {

class Context {
public:
    Context();

    // Adds a backend specified by the name.
    // It returns true if the backend is actually created.
    // If the backend already exists, this function does nothing and returns false.
    bool AddBackend(const std::string& backend_name);

    // Gets the backend specified by the name.
    // If the backend does not exist, this function automatically creates it.
    Backend& GetBackend(const std::string& backend_name);

    // Gets the device specified by the device ID.
    // If the backend and/or device do not exist, this function automatically creates them.
    Device& GetDevice(const DeviceId& device_id);

    // Gets/sets the default device of this context.
    void SetDefaultDevice(Device& device);
    Device& GetDefaultDevice() const;

private:
};

// Gets/sets the context that used by default when current context is not set.
Context& GetDefaultContext();
void SetDefaultContext(Context* context);

// Gets/sets thread local context.
Context& GetCurrentContext();
void SetCurrentContext(Context* context);

}  // namespace xchainer
