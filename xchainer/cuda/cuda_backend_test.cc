#include "xchainer/cuda/cuda_backend.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {
namespace {

TEST(CudaBackendTest, GetDeviceCount) {
    int count;
    CheckError(cudaGetDeviceCount(&count));
    EXPECT_EQ(count, CudaBackend().GetDeviceCount());
}

TEST(CudaBackendTest, GetDevice) {
    CudaBackend backend;
    {
        Device& device = backend.GetDevice(0);
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        Device& device3 = backend.GetDevice(3);
        Device& device2 = backend.GetDevice(2);
        EXPECT_EQ(&backend, &device3.backend());
        EXPECT_EQ(3, device3.index());
        EXPECT_EQ(&backend, &device2.backend());
        EXPECT_EQ(2, device2.index());
    }
    {
        EXPECT_THROW(backend.GetDevice(-1), std::out_of_range);
        EXPECT_THROW(backend.GetDevice(backend.GetDeviceCount() + 1), std::out_of_range);
    }
}

TEST(CudaBackendTest, GetName) { EXPECT_EQ("cuda", cuda::CudaBackend().GetName()); }

}  // namespace
}  // namespace cuda
}  // namespace xchainer
