#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/backward.h"
#include "chainerx/device.h"
#include "chainerx/error.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/native/native_device.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/context_session.h"
#include "chainerx/testing/util.h"

namespace chainerx {
namespace {

// Test configuration class
class TestConfig {
public:
    // backend <0> can transfer from backend <1> to backend <2>
    using Key = std::tuple<int, int, int>;

    TestConfig() {
        set_.insert({
                Key{0, 0, 0},  // backend0 can transfer with itself
                Key{0, 0, 1},  // backend0 can transfer to backend1
                Key{0, 2, 0},  // backend0 can transfer from backend2
                               // backend0 and backend3 are incompatible
        });
    }

    // Returns true if the backend `who` can transfer data from backend `from` to backend `to`
    bool CanTransfer(int who, int from, int to) { return set_.find(std::make_tuple(who, from, to)) != set_.end(); }

    // Return the number of test backends
    int num_backends() { return 4; }

private:
    std::set<Key> set_;
};

// Instantiate the global test configuration
TestConfig g_config;

class TestBackend;

// Test device class
class TestDevice : public native::NativeDevice {
public:
    TestDevice(TestBackend& backend, int index);
};

// Test backend class
class TestBackend : public native::NativeBackend {
public:
    TestBackend(Context& context, int num) : native::NativeBackend{context}, num_{num} {}

    int num() const { return num_; }

    std::string GetName() const override { return "backend" + std::to_string(num_); }

    bool SupportsTransfer(Device& src_device, Device& dst_device) override {
        int src_num = dynamic_cast<TestBackend&>(src_device.backend()).num();
        int dst_num = dynamic_cast<TestBackend&>(dst_device.backend()).num();
        return g_config.CanTransfer(num_, src_num, dst_num);
    }

    int GetDeviceCount() const override { return 1; }

    std::unique_ptr<Device> CreateDevice(int index) override {
        CHAINERX_ASSERT(index == 0);
        return std::make_unique<TestDevice>(*this, index);
    }

private:
    int num_;
};

// TestDevice ctor
TestDevice::TestDevice(TestBackend& backend, int index) : native::NativeDevice{backend, index} {}

// Test fixture for compatible transfer
class ArrayToDeviceCompatibleTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {
protected:
    void SetUp() override {
        context_session_.emplace();

        default_backend_num_ = ::testing::get<0>(GetParam());
        src_backend_num_ = ::testing::get<1>(GetParam());
        dst_backend_num_ = ::testing::get<2>(GetParam());

        backends_.clear();
        Context& context = context_session_->context();
        for (int i = 0; i < g_config.num_backends(); ++i) {
            Backend& backend = context.CreateBackend<TestBackend>("test_backend" + std::to_string(i), i);
            backends_.emplace_back(backend);
        }

        // Set default backend (only if default_backend_num is non-negative)
        if (default_backend_num_ >= 0) {
            device_scope_ = std::make_unique<DeviceScope>(*GetDefaultDevicePtr());
        }
    }

    void TearDown() override {
        device_scope_.reset();
        context_session_.reset();
        backends_.clear();
    }

    Device* GetDefaultDevicePtr() {
        if (default_backend_num_ < 0) {
            return nullptr;
        }
        return &backends_[default_backend_num_].get().GetDevice(0);
    }

    Device& GetSourceDevice() { return backends_[src_backend_num_].get().GetDevice(0); }

    Device& GetDestinationDevice() { return backends_[dst_backend_num_].get().GetDevice(0); }

private:
    absl::optional<testing::ContextSession> context_session_;
    std::unique_ptr<DeviceScope> device_scope_;
    std::vector<std::reference_wrapper<Backend>> backends_;
    int default_backend_num_{};
    int src_backend_num_{};
    int dst_backend_num_{};
};

void ExpectArraysEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    VisitDtype(expected.dtype(), [&expected, &actual](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> expected_iarray{expected};
        IndexableArray<const T> actual_iarray{actual};
        Indexer<> indexer{expected.shape()};

        for (auto it = indexer.It(0); it; ++it) {
            EXPECT_EQ(expected_iarray[it], actual_iarray[it]);
        }
    });
}

TEST_P(ArrayToDeviceCompatibleTest, ToDevice) {
    Device& src_dev = GetSourceDevice();
    Device& dst_dev = GetDestinationDevice();
    Device& default_device = GetDefaultDevice();

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = FromContiguousHostData({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), src_dev);

    // Transfer
    Array b = a.ToDevice(dst_dev);

    EXPECT_EQ(&b.device(), &dst_dev) << "Array::ToDevice must allocate an array on the specified device.";
    EXPECT_EQ(&a.device(), &src_dev) << "Array::ToDevice must not alter the device of the original array.";
    if (&dst_dev == &src_dev) {
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice must return an alias in same-device transfer.";
    }
    EXPECT_EQ(&GetDefaultDevice(), &default_device) << "Array::ToDevice must not alter the default device.";
    ExpectArraysEqual(a, b);
}

TEST_P(ArrayToDeviceCompatibleTest, ToDeviceNonContiguous) {
    Device& src_dev = GetSourceDevice();
    Device& dst_dev = GetDestinationDevice();
    Device& default_device = GetDefaultDevice();

    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>().WithPadding(1).WithDevice(src_dev);

    // Transfer
    Array b = a.ToDevice(dst_dev);

    EXPECT_EQ(&b.device(), &dst_dev) << "Array::ToDevice must allocate an array on the specified device.";
    EXPECT_EQ(&a.device(), &src_dev) << "Array::ToDevice must not alter the device of the original array.";
    if (&dst_dev == &src_dev) {
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice must return an alias in same-device transfer.";
    }
    EXPECT_EQ(&GetDefaultDevice(), &default_device) << "Array::ToDevice must not alter the default device.";
    EXPECT_EQ(&src_dev != &dst_dev, b.IsContiguous()) << "Array::ToDevice must return a contiguous array if device transfer occurs.";
    ExpectArraysEqual(a, b);
}

INSTANTIATE_TEST_CASE_P(
        BackendCombination,
        ArrayToDeviceCompatibleTest,
        ::testing::Values(
                std::make_tuple(-1, 0, 0),  // transfer between same devices
                std::make_tuple(-1, 0, 1),  // transfer to 1
                std::make_tuple(-1, 2, 0),  // transfer from 2
                std::make_tuple(2, 0, 1)));  // checks default device does not change

// Test for incompatible transfer
TEST(ArrayToDeviceIncompatibleTest, ToDeviceIncompatible) {
    testing::ContextSession context_session;
    TestBackend src_backend{context_session.context(), 0};  // incompatible configuration
    TestBackend dst_backend{context_session.context(), 3};

    Device& src_dev = src_backend.GetDevice(0);
    Device& dst_dev = dst_backend.GetDevice(0);

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = FromContiguousHostData({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), src_dev);

    // Transfer
    EXPECT_THROW(a.ToDevice(dst_dev), ChainerxError) << "Array::ToDevice must throw if incompatible device is given.";
}

TEST(ArrayToDeviceArithmeticTest, Arithmetic) {
    CHAINERX_REQUIRE_DEVICE("native", 3);
    testing::ContextSession context_session;
    Backend& backend = context_session.context().CreateBackend<native::NativeBackend>("native_test_backend");

    Device& dev0 = backend.GetDevice(0);
    Device& dev1 = backend.GetDevice(1);
    Device& dev2 = backend.GetDevice(2);  // default device
    DeviceScope device_scope{dev2};

    // Allocate the source array
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    float data0[]{1.0f, 2.0f};
    float data1[]{3.0f, 4.0f};
    float data2[]{5.0f, 6.0f};
    Shape shape{2, 1};
    Array a0 = FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(data0, nop), dev0);
    Array a1 = FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(data1, nop), dev0);
    Array a2 = FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(data2, nop), dev1);
    a0.RequireGrad();
    a1.RequireGrad();
    a2.RequireGrad();

    // Test preconditions
    ASSERT_EQ(&dev0, &a0.device());
    ASSERT_EQ(&dev0, &a1.device());
    ASSERT_EQ(&dev1, &a2.device());

    // Forward
    Array b = a0 * a1;
    Array b_dev1 = b.ToDevice(dev1);
    Array c = b_dev1 + a2;

    ASSERT_TRUE(a0.IsGradRequired());
    ASSERT_TRUE(a1.IsGradRequired());
    ASSERT_TRUE(a2.IsGradRequired());
    ASSERT_FALSE(c.IsGradRequired());
    ASSERT_FALSE(b_dev1.IsGradRequired());
    ASSERT_FALSE(b.IsGradRequired());
    ASSERT_TRUE(testing::IsBackpropIdsEqual({GetDefaultContext().default_backprop_id()}, c));
    ASSERT_TRUE(testing::IsBackpropIdsEqual({GetDefaultContext().default_backprop_id()}, b_dev1));
    ASSERT_TRUE(testing::IsBackpropIdsEqual({GetDefaultContext().default_backprop_id()}, b));

    // Check forward correctness
    EXPECT_EQ(&dev0, &b.device());
    EXPECT_EQ(&dev1, &b_dev1.device());
    EXPECT_EQ(&dev1, &c.device());
    float datay[]{8.0f, 14.0f};  // d0 * d1 + d2
    ExpectArraysEqual(c, FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(datay, nop)));

    // Backward
    Backward(c);

    // Check backward correctness
    ASSERT_TRUE(a0.GetGrad().has_value());
    ASSERT_TRUE(a1.GetGrad().has_value());
    ASSERT_TRUE(a2.GetGrad().has_value());
    EXPECT_EQ(&dev0, &a0.GetGrad()->device());
    EXPECT_EQ(&dev0, &a1.GetGrad()->device());
    EXPECT_EQ(&dev1, &a2.GetGrad()->device());
    float data0_grad[]{3.0f, 4.0f};
    float data1_grad[]{1.0f, 2.0f};
    float data2_grad[]{1.0f, 1.0f};
    ExpectArraysEqual(*a0.GetGrad(), FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(data0_grad, nop)));
    ExpectArraysEqual(*a1.GetGrad(), FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(data1_grad, nop)));
    ExpectArraysEqual(*a2.GetGrad(), FromContiguousHostData(shape, Dtype::kFloat32, std::shared_ptr<float>(data2_grad, nop)));
}

}  // namespace
}  // namespace chainerx
