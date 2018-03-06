#pragma once

#include <mutex>

#include "xchainer/backend.h"

#define XCHAINER_REQUIRE_DEVICE(backend, num)                               \
    do {                                                                    \
        if (xchainer::testing::SkipIfDeviceUnavailable((backend), (num))) { \
            return;                                                         \
        }                                                                   \
    } while (0)

namespace xchainer {
namespace testing {

int GetSkippedNativeTestCount();

int GetSkippedCudaTestCount();

int GetDeviceLimit(Backend& backend);

bool SkipIfDeviceUnavailable(Backend& backend, int num);

}  // namespace testing
}  // namespace xchainer
