#include "xchainer/backprop.h"

#include <vector>

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

std::vector<Array> MakeFullArrays(const Shape& shape, const std::vector<float>& values, bool requires_grad) {
    std::vector<Array> ret;
    for (float value : values) {
        ret.push_back(Array::Full(shape, value));
        ret.back().set_requires_grad(requires_grad);
    }
    return ret;
}

class BackpropTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    template <typename T>
    void ExpectEqual(const Array& expected, const Array& actual) const {
        EXPECT_EQ(expected.dtype(), actual.dtype());
        EXPECT_EQ(expected.shape(), actual.shape());
        ExpectDataEqual<T>(expected, actual);
    }

    template <typename T>
    void ExpectDataEqual(const Array& expected, const Array& actual) const {
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
            EXPECT_EQ(expected_data[i], actual_data[i]);
        }
    }

    // Checks the correctness of Backward() applied to the output of a given function.
    // Gradients are only computed w.r.t. target_inputs, and are compared to expected_grads.
    template <typename Fprop>
    void CheckBackprop(std::vector<Array>& target_inputs, std::vector<Array>& other_inputs, std::vector<Array>& expected_grads,
                       Fprop&& fprop) const {
        EXPECT_EQ(expected_grads.size(), target_inputs.size());
        auto y = fprop(target_inputs, other_inputs);
        Backward(y);
        for (size_t i = 0; i < expected_grads.size(); ++i) {
            ExpectEqual<float>(expected_grads[i], *target_inputs[i].grad());
        }
    }

    // Simple version. It makes and uses an array with one element for each input.
    template <typename Fprop>
    void CheckBackprop(std::vector<float> target_inputs, std::vector<float> other_inputs, std::vector<float> expected_grads,
                       Fprop&& fprop) const {
        auto xs = MakeFullArrays({1}, target_inputs, true);
        auto other_xs = MakeFullArrays({1}, other_inputs, false);
        auto expected_gxs = MakeFullArrays({1}, expected_grads, false);
        CheckBackprop(xs, other_xs, expected_gxs, std::forward<Fprop>(fprop));
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(BackpropTest, Backward) {
    CheckBackprop({2.0f}, {3.0f}, {7.0f}, [](auto& xs, auto& ys) { return xs[0] * (xs[0] + ys[0]); });
}

TEST_P(BackpropTest, DoubleBackprop) {
    auto fprop = [](auto& xs, auto& ys) {
        auto z = xs[0] * (xs[0] + ys[0]);
        Backward(z);
        auto gx = *xs[0].grad();
        xs[0].ClearGrad();
        return gx;
    };
    CheckBackprop({2.0f}, {3.0f}, {2.0f}, fprop);
}

TEST_P(BackpropTest, BackwardIdenticalInputs) {
    CheckBackprop({2.0f}, {}, {2.0f}, [](auto& xs, __attribute__((unused)) auto& ys) { return xs[0] + xs[0]; });
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, BackpropTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                         std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                         std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
