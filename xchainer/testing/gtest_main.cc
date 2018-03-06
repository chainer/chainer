#include <gtest/gtest.h>
#include <iostream>

#include "xchainer/testing/util.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int status = RUN_ALL_TESTS();

    if (xchainer::testing::GetMinSkippedNativeDevice() >= 0) {
        std::cout << "SKIPPED: NATIVE tests requiring devices more than " << xchainer::testing::GetMinSkippedNativeDevice() << "."
                  << std::endl;
    }
    if (xchainer::testing::GetMinSkippedCudaDevice() >= 0) {
        std::cout << "SKIPPED: CUDA tests requiring devices more than " << xchainer::testing::GetMinSkippedCudaDevice() << "." << std::endl;
    }

    return status;
}
