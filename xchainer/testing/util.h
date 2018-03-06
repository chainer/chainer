#pragma once

#include <mutex>

#include "xchainer/backend.h"

#define XCHAINER_REQUIRE_DEVICE(backend, num)                     \
    do {                                                          \
        if (xchainer::testing::RequireDevice((backend), (num))) { \
            return;                                               \
        }                                                         \
    } while (0)

namespace xchainer {
namespace testing {

int GetSkippedNativeTestCount();

int GetSkippedCudaTestCount();

int GetDeviceLimit(Backend& backend);

bool RequireDevice(Backend& backend, int num);

}  // namespace testing
}  // namespace xchainer
