#include <iostream>

#include <gtest/gtest.h>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#include "chainerx/testing/util.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int status = RUN_ALL_TESTS();

    if (chainerx::testing::testing_internal::GetSkippedNativeTestCount() > 0) {
        chainerx::Context ctx;
        chainerx::Backend& backend = ctx.GetBackend("native");
        std::cout << "[  SKIPPED ] " << chainerx::testing::testing_internal::GetSkippedNativeTestCount()
                  << " NATIVE tests requiring devices more than " << chainerx::testing::testing_internal::GetDeviceLimit(backend) << "."
                  << std::endl;
    }
#ifdef CHAINERX_ENABLE_CUDA
    if (chainerx::testing::testing_internal::GetSkippedCudaTestCount() > 0) {
        chainerx::Context ctx;
        chainerx::Backend& backend = ctx.GetBackend("cuda");
        std::cout << "[  SKIPPED ] " << chainerx::testing::testing_internal::GetSkippedCudaTestCount()
                  << " CUDA tests requiring devices more than " << chainerx::testing::testing_internal::GetDeviceLimit(backend) << "."
                  << std::endl;
    }
#endif  // CHAINERX_ENABLE_CUDA

    return status;
}
