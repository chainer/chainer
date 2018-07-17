#pragma once

#include <string>

#include "xchainer/backend.h"

// Skip a test if required number of devices are unavailable.
//
// If `backend` is a string, the global default context must be set in advance.
//
// EXAMPLE:
//
//     TEST(ArrayTest, FooTest) {
//        XCHAINER_REQUIRE_DEVICE(cuda_backend, 2);  // backend instance
//        // XCHAINER_REQUIRE_DEVICE("cuda", 2);  // backend name for default context
//
//        // Write your tests here
//     }
#define XCHAINER_REQUIRE_DEVICE(backend, required_num)                                                 \
    do {                                                                                               \
        if (xchainer::testing::testing_internal::SkipIfDeviceUnavailable((backend), (required_num))) { \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

namespace xchainer {
namespace testing {
namespace testing_internal {

int GetSkippedNativeTestCount();

int GetSkippedCudaTestCount();

int GetDeviceLimit(Backend& backend);

bool SkipIfDeviceUnavailable(Backend& backend, int required_num);

bool SkipIfDeviceUnavailable(const std::string& backend_name, int required_num);

}  // namespace testing_internal
}  // namespace testing
}  // namespace xchainer
