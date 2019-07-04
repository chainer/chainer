#include "chainerx/numerical_gradient.h"

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/backprop_scope.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/check_backward.h"
#include "chainerx/context.h"
#include "chainerx/graph.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/op_node.h"
#include "chainerx/shape.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {
namespace {

using Arrays = std::vector<Array>;
using Fprop = std::function<std::vector<Array>(const std::vector<Array>&)>;

Arrays ForwardWithIncorrectBackward(const Arrays& inputs) {
    const Array& in = inputs[0];
    Array out = EmptyLike(in);

    BackwardBuilder bb{"incorrect_unary", in, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = gout * gout;
        });
    }
    bb.Finalize();

    VisitDtype(in.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> in_iarray{in};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{out.shape()};

        for (auto it = indexer.It(0); it; ++it) {
            out_iarray[it] = in_iarray[it];
        }
    });

    return {out};
}

class CheckBackwardTest : public ::testing::Test {
protected:
    void SetUp() override { device_session_.emplace(DeviceId{native::NativeBackend::kDefaultName, 0}); }

    void TearDown() override { device_session_.reset(); }

protected:
    template <typename T>
    void CheckCheckBackward(
            bool expect_correct,
            const Fprop& fprop,
            const Shape& shape,
            const std::vector<T>& input_data,
            const std::vector<T>& grad_output_data,
            const std::vector<T>& eps_data,
            double atol,
            double rtol,
            const std::string& backprop_name) {
        Array input = testing::BuildArray(shape).WithData(input_data);
        BackpropScope backprop_scope{backprop_name};

        input.RequireGrad(backprop_scope.backprop_id());

        Arrays grad_outputs{testing::BuildArray(shape).WithData(grad_output_data)};
        Arrays eps{testing::BuildArray(shape).WithData(eps_data)};

        if (expect_correct) {
            CheckBackward(fprop, {input}, grad_outputs, eps, 2, atol, rtol, backprop_scope.backprop_id());
        } else {
            EXPECT_THROW(CheckBackward(fprop, {input}, grad_outputs, eps, 2, atol, rtol, backprop_scope.backprop_id()), GradientCheckError);
        }
    }

private:
    absl::optional<testing::DeviceSession> device_session_;
};

class CheckDoubleBackwardTest : public ::testing::Test {
protected:
    void SetUp() override { device_session_.emplace(DeviceId{native::NativeBackend::kDefaultName, 0}); }

    void TearDown() override { device_session_.reset(); }

protected:
    template <typename T>
    void CheckCheckDoubleBackward(
            const Fprop& fprop,
            const Shape& shape,
            const std::vector<T>& input_data,
            const std::vector<T>& grad_output_data,
            const std::vector<T>& grad_grad_input_data,
            const std::vector<T>& eps_input_data,
            const std::vector<T>& eps_grad_output_data,
            double atol,
            double rtol,
            const std::string& backprop_name) {
        Arrays inputs{testing::BuildArray(shape).WithData(input_data)};
        Arrays grad_outputs{testing::BuildArray(shape).WithData(grad_output_data)};
        Arrays grad_grad_inputs{testing::BuildArray(shape).WithData(grad_grad_input_data)};
        Arrays eps{testing::BuildArray(shape).WithData(eps_input_data), testing::BuildArray(shape).WithData(eps_grad_output_data)};
        BackpropScope backprop_scope{backprop_name};

        for (auto& input : inputs) {
            input.RequireGrad(backprop_scope.backprop_id());
        }
        for (auto& grad_output : grad_outputs) {
            grad_output.RequireGrad(backprop_scope.backprop_id());
        }

        // A failure occurs if backward computation and numerical gradients have differences
        CheckDoubleBackwardComputation(fprop, inputs, grad_outputs, grad_grad_inputs, eps, 2, atol, rtol, backprop_scope.backprop_id());
    }

private:
    absl::optional<testing::DeviceSession> device_session_;
};

TEST_F(CheckBackwardTest, CorrectBackward) {
    using T = float;
    std::vector<T> input_data{1.f, 2.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    Fprop fprop = [](const Arrays& inputs) -> Arrays { return {inputs[0] * inputs[0]}; };
    CheckCheckBackward(true, fprop, {1, 3}, input_data, grad_output_data, eps_data, 1e-5, 1e-4, "bp1");
}

TEST_F(CheckBackwardTest, CorrectBackwardWithNonDoubleDifferentiableFunction) {
    using T = float;
    std::vector<T> input_data{1.f, 2.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    Fprop fprop = [](const Arrays& inputs) -> Arrays { return {-inputs[0]}; };
    CheckCheckBackward(true, fprop, {1, 3}, input_data, grad_output_data, eps_data, 1e-5, 1e-4, "bp1");
}

TEST_F(CheckBackwardTest, IncorrectBackward) {
    using T = float;
    std::vector<T> input_data{-2.f, 3.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    CheckCheckBackward(false, &ForwardWithIncorrectBackward, {1, 3}, input_data, grad_output_data, eps_data, 1e-5, 1e-4, "bp1");
}

TEST_F(CheckBackwardTest, IncorrectBackwardIdenticalInputOutput) {
    using T = float;
    std::vector<T> input_data{-2.f, 3.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    CheckCheckBackward(
            false,
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {xs[0].AsType(xs[0].dtype(), false)}; },
            {1, 3},
            input_data,
            grad_output_data,
            eps_data,
            1e-4,
            1e-3,
            "bp1");
}

TEST_F(CheckDoubleBackwardTest, CorrectBackward) {
    using T = float;
    std::vector<T> input_data{1.f, 2.f, 3.f};
    std::vector<T> grad_output_data{1.f, 1.f, 1.f};
    std::vector<T> grad_grad_input_data{1.f, 1.f, 1.f};
    std::vector<T> eps_input_data{1e-3f, 1e-3f, 1e-3f};
    std::vector<T> eps_grad_output_data{1e-3f, 1e-3f, 1e-3f};
    Fprop fprop = [](const Arrays& inputs) -> Arrays { return {inputs[0] * inputs[0]}; };
    CheckCheckDoubleBackward(
            fprop, {1, 3}, input_data, grad_output_data, grad_grad_input_data, eps_input_data, eps_grad_output_data, 1e-4, 1e-3, "bp1");
}

}  // namespace
}  // namespace chainerx
