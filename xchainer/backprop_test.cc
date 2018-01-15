#include "xchainer/backprop.h"

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/backprop.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace {

class BackpropTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    template <typename T>
    void AssertEqual(const Array& expected, const Array& actual) {
        ASSERT_EQ(expected.dtype(), actual.dtype());
        ASSERT_EQ(expected.shape(), actual.shape());
        AssertDataEqual<T>(expected, actual);
    }

    template <typename T>
    void AssertDataEqual(const Array& expected, const Array& actual) {
#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA
        auto total_size = expected.shape().total_size();
        const T* expected_data = static_cast<const T*>(expected.data().get());
        const T* actual_data = static_cast<const T*>(actual.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            ASSERT_EQ(expected_data[i], actual_data[i]);
        }
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(BackpropTest, Backward) {
    Array x = Array::Full({1}, 2.0f);
    x.set_requires_grad(true);

    Array y = Array::Full({1}, 3.0f);

    Array z = x * (x + y);
    Backward(z);

    Array e1 = Array::Full({1}, 7.0f);
    AssertEqual<float>(e1, *x.grad());

    Array gx = *x.grad();
    x.ClearGrad();
    Backward(gx);

    Array e2 = Array::Full({1}, 2.0f);
    AssertEqual<float>(e2, *x.grad());
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, BackpropTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                         std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                         std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
