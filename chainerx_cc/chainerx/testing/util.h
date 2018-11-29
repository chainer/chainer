#pragma once

#include <string>

#include "chainerx/backend.h"

// Skip a test if required number of devices are unavailable.
//
// `backend` can be either a string or a Backend instance.
// If it is a string, a temporary context is created to see the number of devices of that backend.
//
// EXAMPLE:
//
//     TEST(ArrayTest, FooTest) {
//        CHAINERX_REQUIRE_DEVICE(cuda_backend, 2);  // backend instance
//        // CHAINERX_REQUIRE_DEVICE("cuda", 2);  // backend name
//
//        // Write your tests here
//     }
#define CHAINERX_REQUIRE_DEVICE(backend, required_num)                                                 \
    do {                                                                                               \
        if (chainerx::testing::testing_internal::SkipIfDeviceUnavailable((backend), (required_num))) { \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

namespace chainerx {
namespace testing {
namespace testing_internal {

int GetSkippedNativeTestCount();

int GetSkippedCudaTestCount();

int GetDeviceLimit(Backend& backend);

bool SkipIfDeviceUnavailable(Backend& backend, int required_num);

bool SkipIfDeviceUnavailable(const std::string& backend_name, int required_num);

}  // namespace testing_internal
}  // namespace testing
}  // namespace chainerx
