#include "xchainer/gradient_check.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/array_repr.h"
#include "xchainer/device.h"
#include "xchainer/op_node.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace {

class GradientCheckTest : public ::testing::Test {
public:
    using Arrays = std::vector<Array>;

    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return Array::FromBuffer(shape, TypeToDtype<T>, std::move(a));
    }

    template <typename T>
    Array MakeArray(const Shape& shape, const T* data) {
        int64_t size = shape.total_size();
        auto a = std::make_unique<T[]>(size);
        std::copy(data, data + size, a.get());
        return Array::FromBuffer(shape, TypeToDtype<T>, std::move(a));
    }

    template <typename T>
    void CheckElementwiseNumericalGradient(const std::function<Arrays(const Arrays&)>& func, const Arrays& center_inputs,
                                           const Arrays& grad_outputs, const Arrays& eps, const Arrays& expected_grads) {
        size_t nin = center_inputs.size();

        auto checked_func = [&](const Arrays& inputs) -> Arrays {
            EXPECT_EQ(inputs.size(), nin) << "The number of inputs given to the function is wrong";
            for (size_t i = 0; i < center_inputs.size(); ++i) {
                EXPECT_EQ(inputs[i].shape(), center_inputs[i].shape()) << "Shape of inputs given to the function is wrong";
                EXPECT_EQ(inputs[i].dtype(), center_inputs[i].dtype()) << "Dtype of inputs given to the function is wrong";
            }

            return func(inputs);
        };

        Arrays grads = CalculateNumericalGradient(checked_func, center_inputs, grad_outputs, eps);

        EXPECT_EQ(grads.size(), expected_grads.size());
        for (size_t i = 0; i < nin; ++i) {
            auto grads_data = static_cast<const T*>(grads[i].data().get());
            auto expected_grads_data = static_cast<const T*>(expected_grads.at(i).data().get());

            int64_t total_size = grads.at(i).total_size();
            for (int64_t i = 0; i < total_size; ++i) {
                EXPECT_FLOAT_EQ(grads_data[i], expected_grads_data[i]) << "gradient mismatch at i=" << i;
            }
        }
    }
};

TEST_F(GradientCheckTest, NumericalGradientAdd) {
    Shape shape{2, 3};
    float data1[]{1.f, 2.f, -3.f, 4.f, 0.5f, 3.f};
    float data2[]{0.f, 1.3f, 2.f, 3.f, -0.5f, 3.f};
    float eps1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float eps2[]{3.f, -2.f, 3.f, -4.f, 3.2f, 0.9f};
    float grad_output_data[]{1.f, -2.f, 3.f, 0.f, 5.2f, 6.f};

    Arrays inputs = {
        MakeArray(shape, data1), MakeArray(shape, data2),
    };
    Arrays eps = {
        MakeArray(shape, eps1), MakeArray(shape, eps2),
    };
    Arrays grad_outputs = {
        MakeArray(shape, grad_output_data),
    };

    // Forward function
    auto forward = [](const Arrays& inputs) { return Arrays{inputs[0] + inputs[1]}; };

    // Create expected gradients
    Arrays expected_grads = {grad_outputs[0], grad_outputs[0]};

    // Check
    CheckElementwiseNumericalGradient<float>(forward, inputs, grad_outputs, eps, expected_grads);
}

TEST_F(GradientCheckTest, NumericalGradientMul) {
    Shape shape{2, 3};
    float data1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float data2[]{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    float eps1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float eps2[]{3.f, -2.f, 3.f, -4.f, 3.2f, 0.9f};
    float grad_output_data[]{1.f, -2.f, 3.f, 0.f, 5.2f, 6.f};

    Arrays inputs = {
        MakeArray(shape, data1), MakeArray(shape, data2),
    };
    Arrays eps = {
        MakeArray(shape, eps1), MakeArray(shape, eps2),
    };
    Arrays grad_outputs = {
        MakeArray(shape, grad_output_data),
    };

    // Forward function
    auto forward = [](const Arrays& inputs) { return Arrays{inputs[0] * inputs[1]}; };

    // Create expected gradients
    Arrays expected_grads = {inputs[1] * grad_outputs[0], inputs[0] * grad_outputs[0]};

    // Check
    CheckElementwiseNumericalGradient<float>(forward, inputs, grad_outputs, eps, expected_grads);
}

TEST_F(GradientCheckTest, CorretcBackward) {
    Shape shape{2, 3};
    float data1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float data2[]{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    float eps1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float eps2[]{3.f, -2.f, 3.f, -4.f, 3.2f, 0.9f};
    float grad_output_data[]{1.f, -2.f, 3.f, 0.f, 5.2f, 6.f};
    float atol = 1e-5;
    float rtol = 1e-4;

    auto forward = [](const Arrays& inputs) -> Arrays {
        const Array& lhs = inputs[0];
        const Array& rhs = inputs[1];

        bool requires_grad = true;

        Array out = Array::EmptyLike(lhs);
        out.set_requires_grad(requires_grad);

        if (requires_grad) {
            std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
            std::shared_ptr<ArrayNode> out_node = out.RenewNode();

            std::function<Array(const Array&)> empty_func;
            auto lhs_func = requires_grad ? [rhs](const Array& gout) { return gout * rhs; } : empty_func;
            auto rhs_func = requires_grad ? [lhs](const Array& gout) { return gout * lhs; } : empty_func;
            auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
            std::shared_ptr<OpNode> op_node =
                std::make_shared<OpNode>("test_mul", 0, std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node}, backward_functions);
            out_node->set_next_node(op_node);
        }

        VisitDtype(lhs.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            int64_t total_size = lhs.total_size();
            auto* ldata = static_cast<const T*>(lhs.data().get());
            auto* rdata = static_cast<const T*>(rhs.data().get());
            auto* odata = static_cast<T*>(out.data().get());

            for (int64_t i = 0; i < total_size; i++) {
                odata[i] = ldata[i] + rdata[i];
            }
        });

        return {out};
    };
    Arrays inputs = {
        MakeArray(shape, data1), MakeArray(shape, data2),
    };
    Arrays eps = {
        MakeArray(shape, eps1), MakeArray(shape, eps2),
    };
    Arrays grad_outputs = {
        MakeArray(shape, grad_output_data),
    };

    EXPECT_NO_THROW(CheckBackwardComputation(forward, inputs, grad_outputs, eps, atol, rtol));
}

TEST_F(GradientCheckTest, IncorretcBackward) {
  // TODO(hvy): Test an incorrectly implemented backward function
}

}  // namespace
}  // namespace xchainer
