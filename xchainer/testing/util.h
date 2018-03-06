#pragma once

#include <cstdlib>
#include <string>

#include "xchainer/backend.h"

#define SKIP_UNLESS_DEVICE_AVAILABLE(num) \
    if (!IsDeviceAvailable((num))) { return; }

namespace xchainer {
namespace testing {

bool IsDeviceAvailable(int num, Backend& backend = GetDefaultDevice().backend()) {
    int limit = 0;
    const char* env = nullptr;
    if (backend.GetName() == "native") {
        env = std::getenv("XCHAINER_TEST_NATIVE_LIMIT");
    } else if (backend.GetName() == "cuda") {
        env = std::getenv("XCHAINER_TEST_CUDA_LIMIT");
    } else {
        throw BackendError("invalid backend: " + backend.GetName());
    }
    if (env == nullptr) {
        limit = backend.GetDeviceCount();
    }
    else {
        limit = std::stoi(env);
    }
    return num < limit;
}

}  // namespace testing
}  // namespace xchainer

