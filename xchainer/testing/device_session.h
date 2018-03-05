#pragma once

#include <memory>

#include "xchainer/device.h"
#include "xchainer/device_id.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace testing {

class DeviceSession {
public:
    DeviceSession(const DeviceId& device_id) : device_{context_session_.context().GetDevice(device_id)}, device_scope_{device_} {}

    Device& device() { return device_; }

private:
    ContextSession context_session_;
    Device& device_;
    DeviceScope device_scope_;
};

}  // namespace testing
}  // namespace xchainer
