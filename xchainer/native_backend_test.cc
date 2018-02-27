#include "xchainer/native_backend.h"

#include <cstring>

#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/device.h"

namespace xchainer {
namespace {

TEST(NativeBackendTest, GetDeviceCount) {
    Context ctx;
    // TODO(sonots): Get number of CPU cores
    EXPECT_EQ(4, NativeBackend{ctx}.GetDeviceCount());
}

TEST(NativeBackendTest, GetDevice) {
    Context ctx;
    NativeBackend backend{ctx};
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

TEST(NativeBackendTest, GetName) {
    Context ctx;
    EXPECT_EQ("native", NativeBackend{ctx}.GetName());
}

TEST(NativeBackendTest, SupportsTransfer) {
    Context ctx;
    NativeBackend backend{ctx};
    Device& device0 = backend.GetDevice(0);
    Device& device1 = backend.GetDevice(1);

    EXPECT_TRUE(backend.SupportsTransfer(device0, device0));
    EXPECT_TRUE(backend.SupportsTransfer(device0, device1));
}

// Data transfer test
class NativeBackendTransferTest : public ::testing::TestWithParam<::testing::tuple<int, int>> {};

TEST_P(NativeBackendTransferTest, TransferTo) {
    Context ctx;
    NativeBackend backend{ctx};
    Device& device0 = backend.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = backend.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    // TODO(niboshi): Offset is fixed to 0
    std::tuple<std::shared_ptr<void>, size_t> tuple = device0.TransferDataTo(device1, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), std::get<0>(tuple).get(), bytesize));
    // TODO(niboshi): Test offset
}

TEST_P(NativeBackendTransferTest, TransferFrom) {
    Context ctx;
    NativeBackend backend{ctx};
    Device& device0 = backend.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = backend.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    // TODO(niboshi): Offset is fixed to 0
    std::tuple<std::shared_ptr<void>, size_t> tuple = device1.TransferDataFrom(device0, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), std::get<0>(tuple).get(), bytesize));
    // TODO(niboshi): Test offset
}

INSTANTIATE_TEST_CASE_P(Device, NativeBackendTransferTest,
                        ::testing::Values(std::make_tuple(0, 0),  // transfer between same devices
                                          std::make_tuple(0, 1)   // transfer between dfferent native devices
                                          ));

}  // namespace
}  // namespace xchainer
