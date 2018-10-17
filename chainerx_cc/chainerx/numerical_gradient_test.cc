#include "chainerx/numerical_gradient.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_repr.h"
#include "chainerx/context.h"
#include "chainerx/device_id.h"
#include "chainerx/shape.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {
namespace {

class NumericalGradientTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override { device_session_.emplace(DeviceId{GetParam(), 0}); }

    void TearDown() override { device_session_.reset(); }

public:
    using Arrays = std::vector<Array>;

    template <typename T>
    void CheckElementwiseNumericalGradient(
            const std::function<Arrays(const Arrays&)>& func,
            const Arrays& center_inputs,
            const Arrays& grad_outputs,
            const Arrays& eps,
            const Arrays& expected_grads) {
        size_t nin = center_inputs.size();

        std::cout << "check elementwise 1" << std::endl;

        auto checked_func = [&](const Arrays& inputs) -> Arrays {
            EXPECT_EQ(inputs.size(), nin) << "The number of inputs given to the function is wrong";
            for (size_t i = 0; i < center_inputs.size(); ++i) {
                EXPECT_EQ(inputs[i].shape(), center_inputs[i].shape()) << "Shape of inputs given to the function is wrong";
                EXPECT_EQ(inputs[i].dtype(), center_inputs[i].dtype()) << "Dtype of inputs given to the function is wrong";
            }

            return func(inputs);
        };

        std::cout << "check elementwise 2" << std::endl;

        Arrays grads = CalculateNumericalGradient(checked_func, center_inputs, grad_outputs, eps);

        std::cout << "check elementwise 3" << std::endl;

        EXPECT_EQ(grads.size(), expected_grads.size());

        std::cout << "check elementwise 4" << std::endl;

        // GetDefaultDevice().Synchronize();

        std::cout << "check elementwise 5" << std::endl;

        SynchronizeArrays(grads);
        SynchronizeArrays(expected_grads);

        std::cout << "check elementwise 5.1" << std::endl;

        for (size_t i = 0; i < nin; ++i) {
            auto grads_data = static_cast<const T*>(grads[i].data().get());
            std::cout << "check elementwise 5.1.1" << std::endl;
            auto expected_grads_data = static_cast<const T*>(expected_grads.at(i).data().get());
            std::cout << "check elementwise 5.1.2" << std::endl;

            int64_t total_size = grads.at(i).GetTotalSize();
            for (int64_t i = 0; i < total_size; ++i) {
                EXPECT_NEAR(grads_data[i], expected_grads_data[i], 1e-3f) << "gradient mismatch at i=" << i;
            }
            std::cout << "check elementwise 5.1.3" << std::endl;
        }

        std::cout << "check elementwise 6" << std::endl;
    }

private:
    void SynchronizeArrays(const Arrays& arrays) {
        std::for_each(arrays.begin(), arrays.end(), [](const Array& array) { array.device().Synchronize(); });
    }

    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(NumericalGradientTest, NumericalGradientAdd) {
    using T = float;
    Shape shape{2, 3};
    std::vector<T> data1{1.f, 2.f, -3.f, 4.f, 0.5f, 3.f};
    std::vector<T> data2{0.f, 1.3f, 2.f, 3.f, -0.5f, 3.f};
    std::vector<T> eps1{1e-3f, -1e-3f, 1e-3f, 1e-3f, -1e-3f, 1e-3f};
    std::vector<T> eps2{1e-3f, 1e-3f, -1e-3f, 1e-3f, 1e-3f, 1e-3f};
    std::vector<T> grad_output_data{1.f, -2.f, 3.f, 0.f, 3.2f, -1.f};

    Arrays inputs = {
            testing::BuildArray(shape).WithData(data1),
            testing::BuildArray(shape).WithData(data2),
    };
    Arrays eps = {
            testing::BuildArray(shape).WithData(eps1),
            testing::BuildArray(shape).WithData(eps2),
    };
    Arrays grad_outputs = {
            testing::BuildArray(shape).WithData(grad_output_data),
    };

    // Forward function
    auto forward = [](const Arrays& inputs) { return Arrays{inputs[0] + inputs[1]}; };

    // Create expected gradients
    Arrays expected_grads = {grad_outputs[0], grad_outputs[0]};

    // Check
    CheckElementwiseNumericalGradient<float>(forward, inputs, grad_outputs, eps, expected_grads);
}

TEST_P(NumericalGradientTest, NumericalGradientMul) {
    using T = float;
    Shape shape{2, 3};
    std::vector<T> data1{1.f, 2.f, 3.f, 4.f, -2.f, -3.f};
    std::vector<T> data2{0.f, 1.f, 2.f, 3.f, 2.f, 3.f};
    std::vector<T> eps1{1e-3f, -1e-3f, 1e-3f, 1e-3f, -1e-3f, 1e-3f};
    std::vector<T> eps2{1e-3f, 1e-3f, -1e-3f, 1e-3f, 1e-3f, 1e-3f};
    std::vector<T> grad_output_data{1.f, -2.f, 3.f, 0.f, 2.2f, 1.f};

    Arrays inputs = {
            testing::BuildArray(shape).WithData(data1),
            testing::BuildArray(shape).WithData(data2),
    };
    Arrays eps = {
            testing::BuildArray(shape).WithData(eps1),
            testing::BuildArray(shape).WithData(eps2),
    };
    Arrays grad_outputs = {
            testing::BuildArray(shape).WithData(grad_output_data),
    };

    // Forward function
    auto forward = [](const Arrays& inputs) { return Arrays{inputs[0] * inputs[1]}; };

    // Create expected gradients
    Arrays expected_grads = {inputs[1] * grad_outputs[0], inputs[0] * grad_outputs[0]};

    // Check
    CheckElementwiseNumericalGradient<float>(forward, inputs, grad_outputs, eps, expected_grads);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        NumericalGradientTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
