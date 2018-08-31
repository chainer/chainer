#pragma once

#include <memory>

#include "chainerx/device.h"
#include "chainerx/device_id.h"
#include "chainerx/testing/context_session.h"

namespace chainerx {
namespace testing {

class DeviceSession {
public:
    explicit DeviceSession(const DeviceId& device_id) : device_{context_session_.context().GetDevice(device_id)}, device_scope_{device_} {}

    Device& device() { return device_; }

private:
    ContextSession context_session_;
    Device& device_;
    DeviceScope device_scope_;
};

}  // namespace testing
}  // namespace chainerx
