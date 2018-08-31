#pragma once

#include <string>

#include "chainerx/backend.h"

// Skip a test if required number of devices are unavailable.
//
// If `backend` is a string, the global default context must be set in advance.
//
// EXAMPLE:
//
//     TEST(ArrayTest, FooTest) {
//        CHAINERX_REQUIRE_DEVICE(cuda_backend, 2);  // backend instance
//        // CHAINERX_REQUIRE_DEVICE("cuda", 2);  // backend name for default context
//
//        // Write your tests here
//     }

// TODO(imanishi): Do not depend on the default context being set when the backend argument is a string
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
