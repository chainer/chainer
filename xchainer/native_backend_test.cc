#include "xchainer/native_backend.h"

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

}  // namespace
}  // namespace xchainer
