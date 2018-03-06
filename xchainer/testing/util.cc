#include "xchainer/testing/util.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>

#include "xchainer/backend.h"
#include "xchainer/error.h"

namespace xchainer {
namespace testing {

namespace {

std::mutex g_device_available_mutex;
int g_min_skipped_native_device = -1;
int g_min_skipped_cuda_device = -1;

bool IsDeviceAvailable(Backend& backend, int num) {
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
    return num < limit;
}

} // namespace

int GetMinSkippedNativeDevice() { return g_min_skipped_native_device; }

int GetMinSkippedCudaDevice() { return g_min_skipped_cuda_device; }

bool SkipsUnlessDeviceAvailable(Backend& backend, int num) {
    if (IsDeviceAvailable(backend, num)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock{g_device_available_mutex};
        if (backend.GetName() == "native") {
            if (g_min_skipped_native_device < 0) {
                g_min_skipped_native_device = num;
            } else {
                g_min_skipped_native_device = std::min(g_min_skipped_native_device, num);
            }
        } else if (backend.GetName() == "cuda") {
            if (g_min_skipped_cuda_device < 0) {
                g_min_skipped_cuda_device = num;
            } else {
                g_min_skipped_cuda_device = std::min(g_min_skipped_cuda_device, num);
            }
        } else {
            throw BackendError("invalid backend: " + backend.GetName());
        }
    }

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::cout << "[     SKIP ] " << test_info->test_case_name() << "." << test_info->name() << std::endl;
    return true;
}

}  // namespace testing
}  // namespace xchainer
