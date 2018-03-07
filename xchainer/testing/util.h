#pragma once

#include "xchainer/backend.h"

#define XCHAINER_REQUIRE_DEVICE(backend, required_num)                                         \
    do {                                                                                       \
        if (xchainer::testing::internal::SkipIfDeviceUnavailable((backend), (required_num))) { \
            return;                                                                            \
        }                                                                                      \
    } while (0)

namespace xchainer {
namespace testing {
namespace internal {

int GetSkippedNativeTestCount();

int GetSkippedCudaTestCount();

int GetDeviceLimit(Backend& backend);

bool SkipIfDeviceUnavailable(Backend& backend, int required_num);

}  // namespace internal
}  // namespace testing
}  // namespace xchainer
