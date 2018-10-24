#include "chainerx/cuda/cuda_set_device_scope.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "chainerx/error.h"
#include "chainerx/testing/util.h"

namespace chainerx {
namespace cuda {
namespace {

TEST(CudaSetDeviceScopeTest, ScopeSingle) {
    CHAINERX_REQUIRE_DEVICE("cuda", 2);

    ASSERT_TRUE(cudaSetDevice(0) == cudaSuccess);
    int index{0};

    {
        CudaSetDeviceScope scope{1};
        EXPECT_EQ(1, scope.index());
        ASSERT_TRUE(cudaGetDevice(&index) == cudaSuccess);
        EXPECT_EQ(1, index);
    }
    ASSERT_TRUE(cudaGetDevice(&index) == cudaSuccess);
    EXPECT_EQ(0, index);
}

TEST(CudaSetDeviceScopeTest, ScopeMultiple) {
    CHAINERX_REQUIRE_DEVICE("cuda", 2);

    ASSERT_TRUE(cudaSetDevice(0) == cudaSuccess);
    int index{0};

    {
        CudaSetDeviceScope scope1{1};
        EXPECT_EQ(1, scope1.index());
        ASSERT_TRUE(cudaGetDevice(&index) == cudaSuccess);
        EXPECT_EQ(1, index);
        {
            CudaSetDeviceScope scope2{0};
            EXPECT_EQ(0, scope2.index());
            ASSERT_TRUE(cudaGetDevice(&index) == cudaSuccess);
            EXPECT_EQ(0, index);
        }
        ASSERT_TRUE(cudaGetDevice(&index) == cudaSuccess);
        EXPECT_EQ(1, index);
    }
    ASSERT_TRUE(cudaGetDevice(&index) == cudaSuccess);
    EXPECT_EQ(0, index);
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
