#include "chainerx/testing/array.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/shape.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"

namespace chainerx {
namespace testing {
namespace {

TEST(CheckForward, IncorrectOutputNumber) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});
    Array x = Ones({2}, Dtype::kFloat32);

    EXPECT_THROW(
            CheckForward(
                    [](const std::vector<Array>& xs) {
                        return std::vector<Array>{xs[0] * 2, xs[0] + 2};
                    },
                    std::vector<Array>{x},
                    std::vector<Array>{x * 2}),
            testing::RoutinesCheckError);
}

TEST(CheckForward, IncorrectOutputShape) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});
    Array x = Ones({2}, Dtype::kFloat32);

    EXPECT_THROW(
            CheckForward(
                    [](const std::vector<Array>& xs) {
                        return std::vector<Array>{(xs[0] * 2).Reshape({1, 2})};
                    },
                    std::vector<Array>{x},
                    std::vector<Array>{x * 2}),
            testing::RoutinesCheckError);
}

TEST(CheckForward, IncorrectOutputDtype) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});
    Array x = Ones({2}, Dtype::kFloat32);

    EXPECT_THROW(
            CheckForward(
                    [](const std::vector<Array>& xs) { return std::vector<Array>{(xs[0] * 2).AsType(Dtype::kFloat64)}; },
                    std::vector<Array>{x},
                    std::vector<Array>{x * 2}),
            testing::RoutinesCheckError);
}

TEST(CheckForward, IncorrectOutputDevice) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});
    Device& dst_device = device_session.device().backend().GetDevice(1);
    Array x = Ones({2}, Dtype::kFloat32);

    EXPECT_THROW(
            CheckForward(
                    [&dst_device](const std::vector<Array>& xs) { return std::vector<Array>{(xs[0] * 2).ToDevice(dst_device)}; },
                    std::vector<Array>{x},
                    std::vector<Array>{x * 2}),
            testing::RoutinesCheckError);
}

TEST(CheckForward, IncorrectOutputValue) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});
    Array x = Ones({2}, Dtype::kFloat32);

    EXPECT_THROW(
            CheckForward(
                    [](const std::vector<Array>& xs) { return std::vector<Array>{xs[0] * 3}; },
                    std::vector<Array>{x},
                    std::vector<Array>{x * 2}),
            testing::RoutinesCheckError);
}

}  // namespace
}  // namespace testing
}  // namespace chainerx
