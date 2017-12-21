#include "xchainer/cuda/hello.h"

#include <gtest/gtest.h>

#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

TEST(HelloTest, Hello) {
    Device device = GetCurrentDevice();

    SetCurrentDevice("cpu");
    testing::internal::CaptureStdout();
    Hello();
    ASSERT_EQ("Hello, World!\n", testing::internal::GetCapturedStdout());

    SetCurrentDevice("cuda");
    testing::internal::CaptureStdout();
    Hello();
    ASSERT_EQ("Hello, CUDA!\n", testing::internal::GetCapturedStdout());

    SetCurrentDevice(device);
}

}  // namespace cuda
}  // namespace xchainer
