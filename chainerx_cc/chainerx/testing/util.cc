#include "chainerx/testing/util.h"

#include <atomic>
#include <string>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/util.h"

namespace chainerx {
namespace testing {
namespace {

std::atomic<int> g_skipped_native_test_count{0};
std::atomic<int> g_skipped_cuda_test_count{0};

int GetNativeDeviceLimit(Backend& backend) {
    CHAINERX_ASSERT(backend.GetName() == "native");
    static int limit = -1;
    if (limit >= 0) {
        return limit;
    }
    if (absl::optional<std::string> env = GetEnv("CHAINERX_TEST_NATIVE_DEVICE_LIMIT")) {
        try {
            limit = std::stoi(*env);
        } catch (const std::exception&) {
            limit = -1;
        }
        if (limit < 0) {
            throw ChainerxError{"CHAINERX_TEST_NATIVE_DEVICE_LIMIT must be non-negative integer: ", *env};
        }
    } else {
        limit = backend.GetDeviceCount();
    }
    return limit;
}

int GetCudaDeviceLimit(Backend& backend) {
    CHAINERX_ASSERT(backend.GetName() == "cuda");
    static int limit = -1;
    if (limit >= 0) {
        return limit;
    }
    if (absl::optional<std::string> env = GetEnv("CHAINERX_TEST_CUDA_DEVICE_LIMIT")) {
        try {
            limit = std::stoi(*env);
        } catch (const std::exception&) {
            limit = -1;
        }
        if (limit < 0) {
            throw ChainerxError{"CHAINERX_TEST_CUDA_DEVICE_LIMIT must be non-negative integer: ", *env};
        }
    } else {
        limit = backend.GetDeviceCount();
    }
    return limit;
}

}  // namespace

namespace testing_internal {

int GetSkippedNativeTestCount() { return g_skipped_native_test_count; }

int GetSkippedCudaTestCount() { return g_skipped_cuda_test_count; }

int GetDeviceLimit(Backend& backend) {
    if (backend.GetName() == "native") {
        return GetNativeDeviceLimit(backend);
    }
    if (backend.GetName() == "cuda") {
        return GetCudaDeviceLimit(backend);
    }
    throw BackendError{"invalid backend: ", backend.GetName()};
}

bool SkipIfDeviceUnavailable(Backend& backend, int required_num) {
    if (GetDeviceLimit(backend) >= required_num) {
        return false;
    }
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::cout << "[     SKIP ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;

    if (backend.GetName() == "native") {
        ++g_skipped_native_test_count;
    } else if (backend.GetName() == "cuda") {
        ++g_skipped_cuda_test_count;
    } else {
        throw BackendError{"invalid backend: ", backend.GetName()};
    }
    return true;
}

bool SkipIfDeviceUnavailable(const std::string& backend_name, int required_num) {
    return SkipIfDeviceUnavailable(Context{}.GetBackend(backend_name), required_num);
}

}  // namespace testing_internal
}  // namespace testing
}  // namespace chainerx
