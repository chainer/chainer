#include "xchainer/cuda/cuda_backend.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {
namespace {

TEST(CudaBackendTest, GetDeviceCount) {
    Context ctx;
    int count = 0;
    CheckError(cudaGetDeviceCount(&count));
    EXPECT_EQ(count, CudaBackend(ctx).GetDeviceCount());
}

TEST(CudaBackendTest, GetDevice) {
    Context ctx;
    CudaBackend backend{ctx};
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

TEST(CudaBackendTest, GetName) {
    Context ctx;
    EXPECT_EQ("cuda", CudaBackend(ctx).GetName());
}

TEST(CudaBackendTest, SupportsTransfer) {
    Context ctx;
    CudaBackend backend{ctx};
    Device& device0 = backend.GetDevice(0);
    Device& device1 = backend.GetDevice(1);

    EXPECT_TRUE(backend.SupportsTransfer(device0, device0));
    EXPECT_TRUE(backend.SupportsTransfer(device0, device1));
}

// Data transfer test
class CudaBackendTransferTest : public ::testing::TestWithParam<::testing::tuple<int, int>> {};

TEST_P(CudaBackendTransferTest, TransferTo) {
    Context ctx;
    CudaBackend backend{ctx};
    Device& device0 = backend.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = backend.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    // TODO(niboshi): Offset is fixed to 0
    std::tuple<std::shared_ptr<void>, size_t> tuple = device0.TransferDataTo(device1, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), std::get<0>(tuple).get(), bytesize));
    // TODO(niboshi): Test offset

    cudaPointerAttributes attr = {};
    CheckError(cudaPointerGetAttributes(&attr, std::get<0>(tuple).get()));
    EXPECT_TRUE(attr.isManaged);
    EXPECT_EQ(device1.index(), attr.device);
}

TEST_P(CudaBackendTransferTest, TransferFrom) {
    Context ctx;
    CudaBackend backend{ctx};
    Device& device0 = backend.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = backend.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    // TODO(niboshi): Offset is fixed to 0
    std::tuple<std::shared_ptr<void>, size_t> tuple = device1.TransferDataFrom(device0, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), std::get<0>(tuple).get(), bytesize));
    // TODO(niboshi): Test offset

    cudaPointerAttributes attr = {};
    CheckError(cudaPointerGetAttributes(&attr, std::get<0>(tuple).get()));
    EXPECT_TRUE(attr.isManaged);
    EXPECT_EQ(device1.index(), attr.device);
}

INSTANTIATE_TEST_CASE_P(Device, CudaBackendTransferTest, ::testing::Values(std::make_tuple(0, 0),  // transfer between same devices
                                                                           std::make_tuple(0, 1)   // transfer between dfferent CUDA devices
                                                                           ));

}  // namespace
}  // namespace cuda
}  // namespace xchainer
