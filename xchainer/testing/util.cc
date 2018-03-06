#include "xchainer/testing/util.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/error.h"

namespace xchainer {
namespace testing {

std::atomic<int> g_skipped_native_test_count{0};
std::atomic<int> g_skipped_cuda_test_count{0};

int GetSkippedNativeTestCount() { return g_skipped_native_test_count; }

int GetSkippedCudaTestCount() { return g_skipped_cuda_test_count; }

int GetDeviceLimit(Backend& backend) {
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
    } else {
        limit = std::stoi(env);
    }
    return limit;
}

bool SkipsUnlessDeviceAvailable(Backend& backend, int num) {
    if (num < GetDeviceLimit(backend)) {
        return false;
    }

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::cout << "[     SKIP ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;

    if (backend.GetName() == "native") {
        ++g_skipped_native_test_count;
    } else if (backend.GetName() == "cuda") {
        ++g_skipped_cuda_test_count;
    } else {
        throw BackendError("invalid backend: " + backend.GetName());
    }

    return true;
}

}  // namespace testing
}  // namespace xchainer
