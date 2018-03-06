#pragma once

#include "xchainer/backend.h"

#define XCHAINER_REQUIRE_DEVICE(backend, required_num)                               \
    do {                                                                             \
        if (xchainer::testing::SkipIfDeviceUnavailable((backend), (required_num))) { \
            return;                                                                  \
        }                                                                            \
    } while (0)

namespace xchainer {
namespace testing {

int GetSkippedNativeTestCount();

int GetSkippedCudaTestCount();

int GetDeviceLimit(Backend& backend);

bool SkipIfDeviceUnavailable(Backend& backend, int required_num);

}  // namespace testing
}  // namespace xchainer
