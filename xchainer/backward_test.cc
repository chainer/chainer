#include "xchainer/backward.h"

#include <algorithm>
#include <string>
#include <vector>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_body_leak_detection.h"
#include "xchainer/array_node.h"
#include "xchainer/backend.h"
#include "xchainer/backward_builder.h"
#include "xchainer/backward_context.h"
#include "xchainer/check_backward.h"
#include "xchainer/context.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/backprop_scope.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/op_node.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/math.h"
#include "xchainer/shape.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

// Asserts all the array bodies are freed in the leak tracker.
::testing::AssertionResult IsAllArrayBodiesFreed(internal::ArrayBodyLeakTracker& tracker) {
    std::ostringstream os;
    if (tracker.IsAllArrayBodiesFreed(os)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << os.str();
}

TEST(BackwardContextTest, InputGrad) {
    // This test checks if BackwardContext::is_input_grad_required and BackwardContext::input_grad work correctly.
    //
    // (x1) <- [forward] <- (y1 := x1 + x2 + x3)
    // (x2) <-
    // (x3) <-
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    auto forward = [](const Array& x1, const Array& x2, const Array& x3, Array& y1) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        Array x3_c = x3.AsGradStopped();
        y1 = x1_c + x2_c + x3_c;

        BackwardBuilder bb{"func", {x1, x2, x3}, y1};
        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([](BackwardContext& /*bctx*/) { FAIL(); });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget({1, 2});
            bt.Define([x1_c](BackwardContext& bctx) {
                EXPECT_FALSE(bctx.is_input_grad_required(1));
                EXPECT_TRUE(bctx.is_input_grad_required(2));

                // input_grad() should hold the given value even if the input does not require gradient.
                Array dummy1 = FullLike(x1_c, 10);
                Array dummy2 = FullLike(x1_c, 20);
                bctx.input_grad(1) = dummy1;
                EXPECT_EQ(internal::GetArrayBody(dummy1), internal::GetArrayBody(bctx.input_grad(1)));

                bctx.input_grad(2) = dummy2;
                EXPECT_EQ(internal::GetArrayBody(dummy2), internal::GetArrayBody(bctx.input_grad(2)));

                // bctx.input_grad(1) is omitted as it's irrelevant to the test.
                bctx.input_grad(2) = bctx.output_grad(0);
            });
        }
    };

    Array x_value = testing::BuildArray({1}).WithLinearData<double>(3);
    Array x1 = x_value.MakeView().RequireGrad(backprop_id1);
    Array x2 = x_value.MakeView().RequireGrad(backprop_id1);
    Array x3 = x_value.MakeView().RequireGrad(backprop_id2);
    Array expected_x3_grad = OnesLike(x_value, x_value.device());
    Array y1{};
    forward(x1, x2, x3, y1);
    Backward({y1}, backprop_id2);
    testing::ExpectAllClose(expected_x3_grad, *x3.GetGrad(backprop_id2));
}

// TODO(hvy): Separate tests of backprop stack manipulation into another test class/fixture and parameterize the outermost graph over the
// default graph and an explicitly scoped graph. Some tests will become redundant. Remove them.
class BackpropTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        std::string backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

public:
    Array MakeFullArray(const Shape& shape, float value) const { return Full(shape, value); }

    std::vector<Array> MakeFullArrays(const Shape& shape, const std::vector<float>& values) const {
        std::vector<Array> ret;
        ret.reserve(values.size());
        for (float value : values) {
            ret.emplace_back(Full(shape, value));
        }
        return ret;
    }

    template <typename T>
    void ExpectEqual(const Array& expected, const Array& actual) const {
        EXPECT_EQ(expected.dtype(), actual.dtype());
        EXPECT_EQ(expected.shape(), actual.shape());
        ExpectDataEqual<T>(expected, actual);
    }

    template <typename T>
    void ExpectDataEqual(const Array& expected, const Array& actual) const {
#ifdef XCHAINER_ENABLE_CUDA
        std::string backend_name = GetParam();
        if (backend_name == "cuda") {
            cuda::CheckCudaError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA
        auto total_size = expected.shape().GetTotalSize();
        auto expected_data = static_cast<const T*>(expected.data().get());
        auto actual_data = static_cast<const T*>(actual.data().get());
        for (decltype(total_size) i = 0; i < total_size; ++i) {
            EXPECT_EQ(expected_data[i], actual_data[i]);
        }
    }

    void CheckArrayGrad(const Array& a) const {
        ASSERT_TRUE(a.GetGrad().has_value());
        EXPECT_EQ(&a.device(), &a.GetGrad()->device());
    }

    void CheckArrayGrad(const std::vector<Array>& as) const {
        for (const auto& a : as) {
            CheckArrayGrad(a);
        }
    }

    void CallBackward(const Array& a) const { Backward(a); }
    void CallBackward(const std::vector<Array>& a) const { Backward({a.begin(), a.end()}); }

    // Checks the correctness of Backward() applied to the output of a given function.
    // Gradients are only computed w.r.t. target_inputs, and are compared to expected_grads.
    template <typename Fprop, typename... Args>
    void CheckBackpropImpl(std::vector<Array>& target_inputs, std::vector<Array>& expected_grads, Fprop&& fprop, Args&&... args) const {
        ASSERT_EQ(expected_grads.size(), target_inputs.size());

        std::for_each(target_inputs.begin(), target_inputs.end(), [](auto& x) { x.RequireGrad(); });

        // y may be Array or vector<Array>
        auto y = fprop(target_inputs, args...);
        CallBackward(y);
        for (size_t i = 0; i < expected_grads.size(); ++i) {
            auto& target_input = target_inputs[i];
            ASSERT_TRUE(target_input.GetGrad().has_value());
            EXPECT_EQ(&target_input.device(), &target_input.GetGrad()->device());
            ExpectEqual<float>(expected_grads[i], *target_input.GetGrad());
        }
        CheckArrayGrad(y);
    }

    template <typename Fprop>
    void CheckBackprop(std::vector<Array>& target_inputs, std::vector<Array>& expected_grads, Fprop&& fprop) const {
        CheckBackpropImpl(target_inputs, expected_grads, fprop);
    }

    template <typename Fprop>
    void CheckBackpropExtraInputs(
            std::vector<Array>& target_inputs, std::vector<Array>& other_inputs, std::vector<Array>& expected_grads, Fprop&& fprop) const {
        CheckBackpropImpl(target_inputs, expected_grads, fprop, other_inputs);
        for (const Array& other_input : other_inputs) {
            EXPECT_THROW(other_input.GetGrad(), XchainerError);
        }
    }

    // Simple versions. It makes and uses an array with one element for each input.
    template <typename Fprop>
    void CheckBackpropSingleElement(
            const std::vector<float>& target_inputs, const std::vector<float>& expected_grads, Fprop&& fprop) const {
        auto xs = MakeFullArrays({1}, target_inputs);
        auto expected_gxs = MakeFullArrays({1}, expected_grads);
        CheckBackprop(xs, expected_gxs, std::forward<Fprop>(fprop));
    }

    template <typename Fprop>
    void CheckBackpropSingleElementExtraInputs(
            const std::vector<float>& target_inputs,
            const std::vector<float>& other_inputs,
            const std::vector<float>& expected_grads,
            Fprop&& fprop) const {
        auto xs = MakeFullArrays({1}, target_inputs);
        auto other_xs = MakeFullArrays({1}, other_inputs);
        auto expected_gxs = MakeFullArrays({1}, expected_grads);
        CheckBackpropExtraInputs(xs, other_xs, expected_gxs, std::forward<Fprop>(fprop));
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(BackpropTest, BackwardBasic) {
    CheckBackpropSingleElement({3.0f, 2.0f}, {2.0f, 3.0f}, [](auto& xs) { return xs[0] * xs[1]; });
    CheckBackpropSingleElement({3.0f, 2.0f, 4.0f}, {8.0f, 12.0f, 6.0f}, [](auto& xs) { return (xs[0] * xs[1]) * xs[2]; });
    CheckBackpropSingleElement({3.0f, 2.0f}, {12.0f, 9.0f}, [](auto& xs) { return (xs[0] * xs[1]) * xs[0]; });
    CheckBackpropSingleElement({3.0f, 2.0f}, {1.0f, 2.0f}, [](auto& xs) { return (xs[0] + xs[1]) + xs[1]; });
}

TEST_P(BackpropTest, BackwardWithExtraInputs) {
    CheckBackpropSingleElementExtraInputs({2.0f, 3.0f}, {4.0f}, {3.0f, 6.0f}, [](auto& xs, auto& ys) { return xs[1] * (xs[0] + ys[0]); });
    CheckBackpropSingleElementExtraInputs({2.0f}, {4.0f}, {4.0f}, [](auto& xs, auto& ys) { return xs[0] * ys[0]; });
}

TEST_P(BackpropTest, BackwardMultipleOutputs) {
    CheckBackpropSingleElement({2.0f, 3.0f}, {4.0f, 6.0f}, [](auto& xs) -> std::vector<Array> { return {xs[0] * xs[0], xs[1] * xs[1]}; });
    CheckBackpropSingleElement({2.0f, 3.0f}, {4.0f, 3.0f}, [](auto& xs) -> std::vector<Array> { return {xs[0] * xs[1], xs[0] + xs[1]}; });
    CheckBackpropSingleElement({2.0f, 3.0f}, {21.0f, 16.0f}, [](auto& xs) -> std::vector<Array> {
        Array z = xs[0] * xs[1];
        return {xs[0] * z, xs[1] * z};
    });
}

TEST_P(BackpropTest, BackwardWithComplicatedRanks) {
    CheckBackpropSingleElement({1.0f}, {-2.0f}, [](auto& xs) {
        Array a = -xs[0] + 0;
        return -(-a) + a;
    });
}

TEST_P(BackpropTest, TryBackwardFromArrayWithoutNode) {
    auto xs = MakeFullArrays({1}, {2.0f, 3.0f});
    auto y1 = xs[0] * xs[1];  // without graph
    EXPECT_THROW(Backward(y1), XchainerError);
    for (auto& x : xs) {
        x.RequireGrad();
    }
    auto y2 = xs[0] * xs[1];  // with graph
    EXPECT_THROW(Backward({y1, y2}), XchainerError);
}

TEST_P(BackpropTest, BackwardSoleArrayNode) {
    auto x = Full({1}, 2.0f);
    x.RequireGrad();
    Backward(x);
    auto e = OnesLike(x);
    ExpectEqual<float>(e, *x.GetGrad());
}

TEST_P(BackpropTest, DoubleBackprop) {
    auto fprop = [](auto& xs, auto& ys) {
        auto z = xs[0] * (xs[0] + ys[0]);
        Backward(z, nonstd::nullopt, DoubleBackpropOption::kEnable);
        auto gx = *xs[0].GetGrad();  // 2x + y
        xs[0].ClearGrad();
        return gx;
    };
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {2.0f}, fprop);
}

#ifdef XCHAINER_ENABLE_CUDA
TEST_P(BackpropTest, BackpropOnNonDefaultDevice) {
    std::string another_backend = GetParam() == "cuda" ? "native" : "cuda";
    CheckBackpropSingleElement({3.0f, 2.0f}, {2.0f, 3.0f}, [another_backend](auto& xs) {
        auto ret = xs[0] * xs[1];
        // This device switch also affects backward
        SetDefaultDevice(&GetDefaultContext().GetDevice({another_backend, 0}));
        return ret;
    });
}
#endif  // XCHAINER_ENABLE_CUDA

TEST_P(BackpropTest, MultipleGraphsBackprop) {
    BackpropScope backprop_scope_y{"bp_y"};
    BackpropScope backprop_scope_x{"bp_x"};
    BackpropId bp_x = backprop_scope_x.backprop_id();
    BackpropId bp_y = backprop_scope_y.backprop_id();

    Array x_value = Full({1}, 2.0f);
    Array y_value = Full({1}, 3.0f);
    Array x = x_value.MakeView().RequireGrad(bp_x);
    Array y = y_value.MakeView().RequireGrad(bp_y);

    Array z = x * (x + y);

    Backward(z, bp_x, DoubleBackpropOption::kDisable);

    Array gx = *x.GetGrad(bp_x);  // 2x + y
    EXPECT_FALSE(gx.IsGradRequired(bp_x));
    EXPECT_TRUE(gx.IsGradRequired(bp_y));
    testing::ExpectEqual(2 * x_value + y_value, gx);

    Array w = x * gx;
    Backward(w, bp_y, DoubleBackpropOption::kDisable);

    Array gy = *y.GetGrad(bp_y);
    EXPECT_FALSE(gy.IsGradRequired(bp_x));
    EXPECT_FALSE(gy.IsGradRequired(bp_y));
    ExpectEqual<float>(x_value, gy);  // x
}

TEST_P(BackpropTest, MultipleGraphsDoubleBackprop) {
    BackpropScope backprop_scope_y{"bp_y"};
    BackpropScope backprop_scope_x{"bp_x"};
    BackpropId bp_x = backprop_scope_x.backprop_id();
    BackpropId bp_y = backprop_scope_y.backprop_id();

    Array x_value = Full({1}, 2.0f);
    Array y_value = Full({1}, 3.0f);
    Array x = x_value.MakeView().RequireGrad(bp_x);
    Array y = y_value.MakeView().RequireGrad(bp_y);

    Array z = x * (x + y);

    Backward(z, bp_x, DoubleBackpropOption::kEnable);

    Array gx = *x.GetGrad(bp_x);  // 2x + y
    EXPECT_TRUE(gx.IsGradRequired(bp_x));
    EXPECT_TRUE(gx.IsGradRequired(bp_y));
    testing::ExpectEqual(2 * x_value + y_value, gx);

    Array w = x * gx;
    Backward(w, bp_y, DoubleBackpropOption::kDisable);

    Array gy = *y.GetGrad(bp_y);
    EXPECT_FALSE(gy.IsGradRequired(bp_x));  // False because bp_x is inner
    EXPECT_FALSE(gy.IsGradRequired(bp_y));
    ExpectEqual<float>(x_value, gy);  // x
}

TEST_P(BackpropTest, BackwardInputToMultipleOps) {
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {7.0f}, [](auto& xs, auto& ys) { return xs[0] * (xs[0] + ys[0]); });
}

TEST_P(BackpropTest, BackwardIdenticalInputs) {
    CheckBackpropSingleElement({2.0f}, {2.0f}, [](auto& xs) { return xs[0] + xs[0]; });
    CheckBackpropSingleElement({3.0f}, {6.0f}, [](auto& xs) { return xs[0] * xs[0]; });
}

TEST_P(BackpropTest, BackwardIdenticalIntermediateNodes) {
    auto fprop = [](auto& xs) {
        auto y = xs[0] + xs[0];
        return y + y;
    };
    CheckBackpropSingleElement({2.0f}, {4.0f}, fprop);
}

TEST_P(BackpropTest, BackwardGivenInputGrad) {
    auto fprop = [](auto& xs) {
        xs[0].SetGrad(OnesLike(xs[0]));
        return xs[0].Copy();
    };
    CheckBackpropSingleElement({1.0f}, {2.0f}, fprop);
}

TEST_P(BackpropTest, BackwardGivenOutputGrad) {
    auto fprop = [](auto& xs, auto& ys) {
        auto z = xs[0] * ys[0];
        z.SetGrad(FullLike(z, 2.0f));
        return z;
    };
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {6.0f}, fprop);
}

TEST_P(BackpropTest, MultipleGraphsBasic) {
    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id_1 = backprop_scope1.backprop_id();
    BackpropId backprop_id_2 = backprop_scope2.backprop_id();

    x1.RequireGrad(backprop_id_1);
    x2.RequireGrad(backprop_id_2);

    Array y1 = x1 * x2;
    Backward(y1, backprop_id_1);

    Array expected_1 = MakeFullArray({1}, {5.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(backprop_id_1));
    EXPECT_FALSE(x2.GetGrad(backprop_id_2));
}

TEST_P(BackpropTest, MultipleGraphsSameInput) {
    Array x1 = MakeFullArray({1}, {3.0f});

    BackpropScope backprop_scope1{"bp1"};
    BackpropId backprop_id_1 = backprop_scope1.backprop_id();

    x1.RequireGrad(backprop_id_1);

    Array y1 = x1 * x1;
    Backward(y1, backprop_id_1);

    Array expected_1 = MakeFullArray({1}, {6.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(backprop_id_1));

    EXPECT_FALSE(x1.GetGrad(backprop_id_1)->IsGradRequired(backprop_id_1));
}

TEST_P(BackpropTest, MultipleGraphsNonExisting) {
    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id_1 = backprop_scope1.backprop_id();
    BackpropId backprop_id_2 = backprop_scope2.backprop_id();

    x1.RequireGrad(backprop_id_1);
    x2.RequireGrad(backprop_id_1);

    Array y1 = x1 * x2;
    EXPECT_THROW(Backward(y1, backprop_id_2), XchainerError);
}

TEST_P(BackpropTest, MultipleGraphsReuseWithDefaultGraph) {
    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();

    x1.RequireGrad(backprop_id);
    x2.RequireGrad();

    Array y1 = x1 * x2;
    Backward(y1, backprop_id);

    Array expected_1 = MakeFullArray({1}, {5.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(backprop_id));
    EXPECT_FALSE(x2.GetGrad());

    x1.ClearGrad(backprop_id);
    x2.ClearGrad();

    Array y2 = x1 * x2;
    Backward(y2);

    Array expected_2 = MakeFullArray({1}, {2.0f});
    ExpectEqual<float>(expected_2, *x2.GetGrad());
    EXPECT_FALSE(x1.GetGrad(backprop_id));

    x1.ClearGrad(backprop_id);
    x2.ClearGrad();

    x1.RequireGrad();
    x2.RequireGrad(backprop_id);

    Array y3 = x1 * x2;
    Backward(y3);

    ExpectEqual<float>(expected_1, *x1.GetGrad());
    ExpectEqual<float>(expected_2, *x2.GetGrad());
    EXPECT_FALSE(x1.GetGrad(backprop_id));
    EXPECT_FALSE(x2.GetGrad(backprop_id));
}

TEST_P(BackpropTest, MultipleGraphsReuse) {
    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    BackpropScope backprop_scope_outer{"bp_outer"};
    BackpropScope backprop_scope_inner{"bp_inner"};
    BackpropId backprop_id_outer = backprop_scope_outer.backprop_id();
    BackpropId backprop_id_inner = backprop_scope_inner.backprop_id();

    x1.RequireGrad(backprop_id_inner);
    x2.RequireGrad(backprop_id_outer);

    Array y1 = x1 * x2;
    Backward(y1, backprop_id_inner);

    Array expected_1 = MakeFullArray({1}, {5.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(backprop_id_inner));
    EXPECT_FALSE(x2.GetGrad(backprop_id_outer));

    x1.ClearGrad(backprop_id_inner);
    x2.ClearGrad(backprop_id_outer);

    Array y2 = x1 * x2;
    Backward(y2, backprop_id_outer);

    Array expected_2 = MakeFullArray({1}, {2.0f});
    ExpectEqual<float>(expected_2, *x2.GetGrad(backprop_id_outer));
    EXPECT_FALSE(x1.GetGrad(backprop_id_inner));

    x1.ClearGrad(backprop_id_inner);
    x2.ClearGrad(backprop_id_outer);

    x1.RequireGrad(backprop_id_outer);
    x2.RequireGrad(backprop_id_inner);

    Array y3 = x1 * x2;
    Backward(y3, backprop_id_outer);

    ExpectEqual<float>(expected_1, *x1.GetGrad(backprop_id_outer));
    ExpectEqual<float>(expected_2, *x2.GetGrad(backprop_id_outer));
    EXPECT_FALSE(x1.GetGrad(backprop_id_inner));
    EXPECT_FALSE(x2.GetGrad(backprop_id_inner));
}

TEST_P(BackpropTest, BackwardDefaultGraphAfterInnerGraph) {
    Array x = MakeFullArray({1}, {2.0f});
    x.RequireGrad();

    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();

    x.RequireGrad(backprop_id);

    Array y = x * x;

    Backward(y, backprop_id);

    EXPECT_TRUE(x.GetGrad(backprop_id)->IsGradRequired());

    Backward(y);  // no throw
}

TEST_P(BackpropTest, BackwardInnerGraphAfterDefaultGraph) {
    Array x = MakeFullArray({1}, {2.0f});
    x.RequireGrad();

    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();

    x.RequireGrad(backprop_id);

    Array y = x * x;

    Backward(y);

    EXPECT_FALSE(x.GetGrad()->IsGradRequired(backprop_id));

    EXPECT_THROW(Backward(y, backprop_id), XchainerError);
}

TEST_P(BackpropTest, BackwardInnerGraphAfterOuterGraph) {
    Array x = MakeFullArray({1}, {2.0f});

    BackpropScope backprop_scope_outer{"bp_outer"};
    BackpropScope backprop_scope_inner{"bp_inner"};
    BackpropId backprop_id_outer = backprop_scope_outer.backprop_id();
    BackpropId backprop_id_inner = backprop_scope_inner.backprop_id();

    x.RequireGrad(backprop_id_outer);
    x.RequireGrad(backprop_id_inner);

    Array y = x * x;

    Backward(y, backprop_id_outer);

    EXPECT_FALSE(x.GetGrad(backprop_id_outer)->IsGradRequired(backprop_id_inner));

    EXPECT_THROW(Backward(y, backprop_id_inner), XchainerError);
}

TEST_P(BackpropTest, BackwardThreeGraphsIncludingDefaultGraph) {
    Array x = MakeFullArray({1}, {2.0f});
    Array y;

    BackpropScope backprop_scope_1{"bp1"};
    BackpropId backprop_id_1 = backprop_scope_1.backprop_id();
    {
        BackpropScope backprop_scope_2{"bp2"};
        BackpropId backprop_id_2 = backprop_scope_2.backprop_id();

        x.RequireGrad();
        x.RequireGrad(backprop_id_1);
        x.RequireGrad(backprop_id_2);

        y = x * x;

        Backward(y, backprop_id_2);

        EXPECT_TRUE(x.GetGrad(backprop_id_2)->IsGradRequired());
        EXPECT_TRUE(x.GetGrad(backprop_id_2)->IsGradRequired(backprop_id_1));

        Backward(y);

        EXPECT_FALSE(x.GetGrad()->IsGradRequired(backprop_id_1));
        EXPECT_FALSE(x.GetGrad()->IsGradRequired(backprop_id_2));
    }

    // Default graph backward is already finished in a deeper scope.
    EXPECT_THROW(Backward(y, backprop_id_1), XchainerError);
}

TEST_P(BackpropTest, BackwardThreeGraphs) {
    Array x = MakeFullArray({1}, {2.0f});
    Array y;

    BackpropScope backprop_scope_1{"bp1"};
    BackpropScope backprop_scope_2{"bp2"};
    BackpropId backprop_id_1 = backprop_scope_1.backprop_id();
    BackpropId backprop_id_2 = backprop_scope_2.backprop_id();
    {
        BackpropScope backprop_scope_3{"bp3"};
        BackpropId backprop_id_3 = backprop_scope_3.backprop_id();

        x.RequireGrad(backprop_id_1);
        x.RequireGrad(backprop_id_2);
        x.RequireGrad(backprop_id_3);

        y = x * x;

        Backward(y, backprop_id_3);

        EXPECT_TRUE(x.GetGrad(backprop_id_3)->IsGradRequired(backprop_id_1));
        EXPECT_TRUE(x.GetGrad(backprop_id_3)->IsGradRequired(backprop_id_2));

        Backward(y, backprop_id_1);

        EXPECT_FALSE(x.GetGrad(backprop_id_1)->IsGradRequired(backprop_id_2));
        EXPECT_FALSE(x.GetGrad(backprop_id_1)->IsGradRequired(backprop_id_3));
    }

    // Outer scope graph backward is already finished in a deeper scope.
    EXPECT_THROW(Backward(y, backprop_id_2), XchainerError);
}

TEST_P(BackpropTest, NoCyclicReferenceInvolvingInputGrad) {
    // This test checks cyclic reference is not formed when the input gradient references the input array.
    // The cycle could happen if input array nodes directly owned their gradients.

    std::weak_ptr<internal::ArrayBody> x_grad_body{};

    {
        BackpropScope backprop_scope{"bp"};
        BackpropId backprop_id = backprop_scope.backprop_id();

        auto forward = [](const Array& x, Array& y) {
            y = x.AsGradStopped() * x.AsGradStopped();

            BackwardBuilder bb{"func", x, y};
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([x](BackwardContext& bctx) {
                // Create an input grad which references the input array.
                bctx.input_grad() = 2 * x * bctx.output_grad();
            });
        };

        Array x = testing::BuildArray({1}).WithLinearData<float>();
        Array y{};

        x.RequireGrad(backprop_id);
        forward(x, y);

        Backward(y, backprop_id, DoubleBackpropOption::kEnable);

        x_grad_body = internal::GetArrayBody(*x.GetGrad(backprop_id));  // Keep weak pointer to the body of x.grad
    }

    // The body of x.grad must have been released.
    EXPECT_EQ(nullptr, x_grad_body.lock());
}

TEST_P(BackpropTest, SomeOfPreviousArrayNodesAreGone) {
    // This test checks the backward of a multiple-output function where some of the output arrays are gone.
    //
    // (x) <- [forward] <- (y1 := (x-1) exp x) <- [view] <- (z1)
    //                  <- (y2 :=     2 exp x)
    //                  <- (y3 :=     3 exp x)
    //                  <- (y4 :=     4 exp x)
    //
    // y1 is kept alive via z1 but other y's are not.

    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    auto forward = [](const Array& x, Array& y1, Array& y2, Array& y3, Array& y4) {
        Array x_const = x.AsGradStopped();

        y1 = Exp(x_const) * (x_const - 1);
        y2 = Exp(x_const) * 2;
        y3 = Exp(x_const) * 3;
        y4 = Exp(x_const) * 4;

        BackwardBuilder bb{"func", x, {y1, y2, y3, y4}};
        BackwardBuilder::Target bt = bb.CreateTarget(0);
        bt.Define([x](BackwardContext& bctx) {
            Array gy1gx = bctx.output_grad(0) * Exp(x) * x;
            Array gy2gx = bctx.output_grad(1) * Exp(x) * 2;
            Array gy3gx = bctx.output_grad(2) * Exp(x) * 3;
            Array gy4gx = bctx.output_grad(3) * Exp(x) * 4;
            bctx.input_grad() = gy1gx + gy2gx + gy3gx + gy4gx;
        });
    };

    Array z1{};
    Array x_value = testing::BuildArray({2, 3}).WithLinearData<float>();
    Array x = x_value.MakeView().RequireGrad();
    {
        Array y1{};
        Array y2{};
        Array y3{};
        Array y4{};
        forward(x, y1, y2, y3, y4);
        z1 = y1.MakeView();
    }
    Backward(z1, nonstd::nullopt);

    Array expected_x_grad = x_value * Exp(x_value);
    testing::ExpectAllClose(expected_x_grad, *x.GetGrad(), 1e-5, 1e-8);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        BackpropTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

class BackpropFunctionTest : public ::testing::TestWithParam<DoubleBackpropOption> {};

TEST_P(BackpropFunctionTest, OneToOneFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gx1_value = 2 * gy1_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, double_backprop_opt, &backprop_id](const Array& x1, Array& y1) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", x1, y1};
            EXPECT_FALSE(bb.is_complete());

            BackwardBuilder::Target bt = bb.CreateTarget(0);
            EXPECT_TRUE(bt.is_definition_required());
            EXPECT_TRUE(static_cast<bool>(bt));
            EXPECT_TRUE(bb.is_complete());

            bt.Define([gy1_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad();  // omit index
                testing::ExpectEqual(gy1_value, gy1);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(backprop_id));
                } else {
                    EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                }
                bctx.input_grad() = 2 * gy1;  // omit index
            });
            EXPECT_TRUE(bb.is_complete());
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    forward(x1, y1);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(backprop_id), backprop_id);
    } else {
        y1.SetGrad(gy1_value, backprop_id);
    }
    Backward({y1}, backprop_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(backprop_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(backprop_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(backprop_id)->IsGradRequired(backprop_id));
    } else {
        EXPECT_FALSE(y1.GetGrad(backprop_id)->IsGradRequired(AnyGraph{}));
    }
}

TEST_P(BackpropFunctionTest, OneToMultiFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gy2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array gx1_value = 2 * gy1_value + 3 * gy2_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, gy2_value, double_backprop_opt, &backprop_id](const Array& x1, Array& y1, Array& y2) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 1;
        y2 = 3 * x1.AsGradStopped() + 2;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));
        ASSERT_FALSE(y2.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", x1, {y1, y2}};
            EXPECT_FALSE(bb.is_complete());

            BackwardBuilder::Target bt = bb.CreateTarget(0);
            EXPECT_TRUE(bt.is_definition_required());
            EXPECT_TRUE(static_cast<bool>(bt));
            EXPECT_TRUE(bb.is_complete());

            bt.Define([gy1_value, gy2_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);  // by index
                const Array& gy2 = bctx.output_grad(1);
                testing::ExpectEqual(gy1_value, gy1);
                testing::ExpectEqual(gy2_value, gy2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(backprop_id));
                    EXPECT_TRUE(gy2.IsGradRequired(backprop_id));
                } else {
                    EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                    EXPECT_FALSE(gy2.IsGradRequired(AnyGraph{}));
                }
                bctx.input_grad(0) = 2 * gy1 + 3 * gy2;  // by index
            });
            EXPECT_TRUE(bb.is_complete());
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    Array y2{};
    forward(x1, y1, y2);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(backprop_id), backprop_id);
        y2.SetGrad(gy2_value.MakeView().RequireGrad(backprop_id), backprop_id);
    } else {
        y1.SetGrad(gy1_value, backprop_id);
        y2.SetGrad(gy2_value, backprop_id);
    }
    Backward({y1, y2}, backprop_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(backprop_id));
    testing::ExpectEqual(gy2_value, *y2.GetGrad(backprop_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(backprop_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(backprop_id)->IsGradRequired(backprop_id));
        EXPECT_TRUE(y2.GetGrad(backprop_id)->IsGradRequired(backprop_id));
    } else {
        EXPECT_FALSE(y1.GetGrad(backprop_id)->IsGradRequired(AnyGraph{}));
        EXPECT_FALSE(y2.GetGrad(backprop_id)->IsGradRequired(AnyGraph{}));
    }
}

TEST_P(BackpropFunctionTest, MultiToOneFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array x2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array x3_value = testing::BuildArray(shape).WithData<T>({-1, 3});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gx1_value = 2 * gy1_value;
    Array gx2_value = 3 * gy1_value;
    Array gx3_value = 1 * gy1_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, double_backprop_opt, &backprop_id](const Array& x1, const Array& x2, const Array& x3, Array& y1) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        ASSERT_TRUE(x2.IsGradRequired(AnyGraph{}));
        ASSERT_TRUE(x3.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 3 * x2.AsGradStopped() + x3.AsGradStopped() + 1;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", {x1, x2, x3}, y1};
            EXPECT_FALSE(bb.is_complete());
            {
                BackwardBuilder::Target bt = bb.CreateTarget(0);
                EXPECT_TRUE(bt.is_definition_required());
                EXPECT_TRUE(static_cast<bool>(bt));
                EXPECT_FALSE(bb.is_complete());

                bt.Define([gy1_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                    const Array& gy1 = bctx.output_grad();  // omit index
                    testing::ExpectEqual(gy1_value, gy1);
                    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                        EXPECT_TRUE(gy1.IsGradRequired(backprop_id));
                    } else {
                        EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                    }

                    // input_grad has null array
                    EXPECT_EQ(nullptr, internal::GetArrayBody(bctx.input_grad()));

                    // input_grad setter
                    bctx.input_grad() = 2 * gy1;  // omit index

                    // Check bctx.input_grad() as a getter
                    Array gx1_back = bctx.input_grad();
                    testing::ExpectEqual(2 * gy1, gx1_back);
                });
            }
            {
                BackwardBuilder::Target bt = bb.CreateTarget({1, 2});
                EXPECT_TRUE(bt.is_definition_required());
                EXPECT_TRUE(static_cast<bool>(bt));
                EXPECT_TRUE(bb.is_complete());

                bt.Define([gy1_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                    const Array& gy1 = bctx.output_grad(0);  // by index
                    testing::ExpectEqual(gy1_value, gy1);
                    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                        EXPECT_TRUE(gy1.IsGradRequired(backprop_id));
                    } else {
                        EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                    }

                    // input_grad has null array
                    EXPECT_EQ(nullptr, internal::GetArrayBody(bctx.input_grad(1)));
                    EXPECT_EQ(nullptr, internal::GetArrayBody(bctx.input_grad(2)));

                    // input_grad setter
                    bctx.input_grad(1) = 3 * gy1;  // by index
                    bctx.input_grad(2) = 1 * gy1;

                    // Check bctx.input_grad() as a getter
                    Array gx2_back = bctx.input_grad(1);
                    Array gx3_back = bctx.input_grad(2);
                    testing::ExpectEqual(3 * gy1, gx2_back);
                    testing::ExpectEqual(1 * gy1, gx3_back);
                });
            }
            EXPECT_TRUE(bb.is_complete());
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array x2 = x2_value.MakeView().RequireGrad(backprop_id);
    Array x3 = x3_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    forward(x1, x2, x3, y1);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(backprop_id), backprop_id);
    } else {
        y1.SetGrad(gy1_value, backprop_id);
    }
    Backward({y1}, backprop_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(backprop_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(backprop_id));
    testing::ExpectEqual(gx2_value, *x2.GetGrad(backprop_id));
    testing::ExpectEqual(gx3_value, *x3.GetGrad(backprop_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(backprop_id)->IsGradRequired(backprop_id));
    } else {
        EXPECT_FALSE(y1.GetGrad(backprop_id)->IsGradRequired(AnyGraph{}));
    }
}

TEST_P(BackpropFunctionTest, MultiToMultiFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array x2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array x3_value = testing::BuildArray(shape).WithData<T>({-1, 3});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gy2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array gx1_value = 2 * gy1_value + 3 * gy2_value;
    Array gx2_value = 3 * gy1_value + 1 * gy2_value;
    Array gx3_value = 0 * gy1_value + 2 * gy2_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, gy2_value, double_backprop_opt, &backprop_id](
                           const Array& x1, const Array& x2, const Array& x3, Array& y1, Array& y2) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        ASSERT_TRUE(x2.IsGradRequired(AnyGraph{}));
        ASSERT_TRUE(x3.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 3 * x2.AsGradStopped() + 1;
        y2 = 3 * x1.AsGradStopped() + 1 * x2.AsGradStopped() + 2 * x3.AsGradStopped() + 4;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));
        ASSERT_FALSE(y2.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", {x1, x2, x3}, {y1, y2}};
            EXPECT_FALSE(bb.is_complete());
            {
                BackwardBuilder::Target bt = bb.CreateTarget(0);
                EXPECT_TRUE(bt.is_definition_required());
                EXPECT_TRUE(static_cast<bool>(bt));
                EXPECT_FALSE(bb.is_complete());

                bt.Define([gy1_value, gy2_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                    const Array& gy1 = bctx.output_grad(0);  // by index
                    const Array& gy2 = bctx.output_grad(1);
                    testing::ExpectEqual(gy1_value, gy1);
                    testing::ExpectEqual(gy2_value, gy2);
                    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                        EXPECT_TRUE(gy1.IsGradRequired(backprop_id));
                        EXPECT_TRUE(gy2.IsGradRequired(backprop_id));
                    } else {
                        EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                        EXPECT_FALSE(gy2.IsGradRequired(AnyGraph{}));
                    }
                    bctx.input_grad(0) = 2 * gy1 + 3 * gy2;  // by index
                });
            }
            {
                BackwardBuilder::Target bt = bb.CreateTarget({1, 2});
                EXPECT_TRUE(bt.is_definition_required());
                EXPECT_TRUE(static_cast<bool>(bt));
                EXPECT_TRUE(bb.is_complete());

                bt.Define([gy1_value, gy2_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                    const Array& gy1 = bctx.output_grad(0);  // by index
                    const Array& gy2 = bctx.output_grad(1);
                    testing::ExpectEqual(gy1_value, gy1);
                    testing::ExpectEqual(gy2_value, gy2);
                    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                        EXPECT_TRUE(gy1.IsGradRequired(backprop_id));
                        EXPECT_TRUE(gy2.IsGradRequired(backprop_id));
                    } else {
                        EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                        EXPECT_FALSE(gy2.IsGradRequired(AnyGraph{}));
                    }

                    Array gx2 = 3 * gy1 + gy2;
                    Array gx3 = 2 * gy2;
                    bctx.input_grad(1) = gx2;  // by index, from non-temporary
                    bctx.input_grad(2) = gx3;
                });
            }
            EXPECT_TRUE(bb.is_complete());
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array x2 = x2_value.MakeView().RequireGrad(backprop_id);
    Array x3 = x3_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    Array y2{};
    forward(x1, x2, x3, y1, y2);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(backprop_id), backprop_id);
        y2.SetGrad(gy2_value.MakeView().RequireGrad(backprop_id), backprop_id);
    } else {
        y1.SetGrad(gy1_value, backprop_id);
        y2.SetGrad(gy2_value, backprop_id);
    }
    Backward({y1, y2}, backprop_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(backprop_id));
    testing::ExpectEqual(gy2_value, *y2.GetGrad(backprop_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(backprop_id));
    testing::ExpectEqual(gx2_value, *x2.GetGrad(backprop_id));
    testing::ExpectEqual(gx3_value, *x3.GetGrad(backprop_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(backprop_id)->IsGradRequired(backprop_id));
        EXPECT_TRUE(y2.GetGrad(backprop_id)->IsGradRequired(backprop_id));
    } else {
        EXPECT_FALSE(y1.GetGrad(backprop_id)->IsGradRequired(AnyGraph{}));
        EXPECT_FALSE(y2.GetGrad(backprop_id)->IsGradRequired(AnyGraph{}));
    }
}

TEST_P(BackpropFunctionTest, SomeInputDoesNotRequireGrad) {
    // This test checks that even if some of input arrays (x1) do not require the gradients, assignment of the input gradients works fine.
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array x2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gx2_value = FullLike(x1_value, 3);

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [](const Array& x1, const Array& x2, Array& y1) {
        y1 = 2 * x1.AsGradStopped() + 3 * x2.AsGradStopped() + 1;
        {
            BackwardBuilder bb{"func", {x1, x2}, y1};
            EXPECT_FALSE(bb.is_complete());

            BackwardBuilder::Target bt = bb.CreateTarget({0, 1});
            EXPECT_TRUE(bt.is_definition_required());
            EXPECT_TRUE(static_cast<bool>(bt));
            EXPECT_TRUE(bb.is_complete());

            bt.Define([](BackwardContext& bctx) {
                EXPECT_FALSE(bctx.is_input_grad_required(0));
                EXPECT_TRUE(bctx.is_input_grad_required(1));
                Array gy1gx1 = 2 * bctx.output_grad();
                Array gy1gx2 = 3 * bctx.output_grad();

                // Input grad can be assigned even if it's not required.
                bctx.input_grad(0) = gy1gx1;
                bctx.input_grad(1) = gy1gx2;

                // The return of `input_grad_required` should not change even after assignment.
                EXPECT_FALSE(bctx.is_input_grad_required(0));
                EXPECT_TRUE(bctx.is_input_grad_required(1));

                // The value of `input_grad` should be retained, even if it's not required.
                testing::ExpectEqual(bctx.input_grad(0), gy1gx1);
                testing::ExpectEqual(bctx.input_grad(1), gy1gx2);
            });
            EXPECT_TRUE(bb.is_complete());
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id2);  // Grad not required for backprop_id1
    Array x2 = x2_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
    Array y1{};
    forward(x1, x2, y1);

    Backward({y1}, backprop_id1, double_backprop_opt);

    testing::ExpectEqual(gx2_value, *x2.GetGrad(backprop_id1));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        // TODO(niboshi): Enable this check
        // EXPECT_TRUE(x2.GetGrad(backprop_id1)->IsGradRequired(backprop_id1));
    } else {
        EXPECT_FALSE(x2.GetGrad(backprop_id1)->IsGradRequired(AnyGraph{}));
    }
    EXPECT_THROW({ x1.GetGrad(backprop_id1); }, XchainerError);
}

TEST_P(BackpropFunctionTest, SomeOutputGradsAreAbsentWhileArrayNodesAreAlive) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array gx1_value = 3 * gy2_value;  // gy1 is ignored

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy2_value, double_backprop_opt, &backprop_id](const Array& x1, Array& y1, Array& y2) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 1;
        y2 = 3 * x1.AsGradStopped() + 2;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));
        ASSERT_FALSE(y2.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", x1, {y1, y2}};
            EXPECT_FALSE(bb.is_complete());

            BackwardBuilder::Target bt = bb.CreateTarget(0);
            EXPECT_TRUE(bt.is_definition_required());
            EXPECT_TRUE(static_cast<bool>(bt));
            EXPECT_TRUE(bb.is_complete());

            bt.Define([gy2_value, double_backprop_opt, &backprop_id](BackwardContext& bctx) {
                EXPECT_FALSE(bctx.HasOutputGrad(0));
                EXPECT_TRUE(bctx.HasOutputGrad(1));

                const Array& gy1 = bctx.output_grad(0);
                const Array& gy2 = bctx.output_grad(1);
                testing::ExpectEqual(ZerosLike(gy2_value), gy1);
                testing::ExpectEqual(gy2_value, gy2);

                EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy2.IsGradRequired(backprop_id));
                } else {
                    EXPECT_FALSE(gy2.IsGradRequired(AnyGraph{}));
                }

                bctx.input_grad() = 2 * gy1 + 3 * gy2;
            });
            EXPECT_TRUE(bb.is_complete());
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    Array y2{};
    forward(x1, y1, y2);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y2.SetGrad(gy2_value.MakeView().RequireGrad(backprop_id), backprop_id);
    } else {
        y2.SetGrad(gy2_value, backprop_id);
    }
    // Start backprop from y2. y1 is ignored.
    Backward(y2, backprop_id, double_backprop_opt);

    testing::ExpectEqual(gy2_value, *y2.GetGrad(backprop_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(backprop_id));
}

INSTANTIATE_TEST_CASE_P(Params, BackpropFunctionTest, ::testing::Values(DoubleBackpropOption::kDisable, DoubleBackpropOption::kEnable));

class BackpropRetainOutputTest : public ::testing::TestWithParam<DoubleBackpropOption> {};

TEST_P(BackpropRetainOutputTest, RetainOutput_OriginalBodyIsAlive) {
    // This test checks retained output array can be retrieved, where its array body (y) is kept alive at the moment of retrieval.
    //
    // (x1) <- [forward] <- (y1 := exp(x1 + 2 x2) + exp(2 x1 + x2))
    // (x2) <-           <- (y2 := exp(x1 + 2 x2) - exp(2 x1 + x2))
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    std::weak_ptr<internal::ArrayBody> y1_body{};
    std::weak_ptr<internal::ArrayBody> y2_body{};

    auto forward = [&backprop_id1, &backprop_id2, &y1_body, &y2_body, double_backprop_opt](
                           const Array& x1, const Array& x2, Array& y1, Array& y2) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = Exp(x1_c + 2 * x2_c) + Exp(2 * x1_c + x2_c);
        y2 = Exp(x1_c + 2 * x2_c) - Exp(2 * x1_c + x2_c);

        Array y1_value = y1.MakeView();
        Array y2_value = y2.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, {y1, y2}};

        // y1 is retrieved with copied tokens.
        // y2 is retrieved with sperarately acquired tokens.
        RetainedOutputToken tok1 = bb.RetainOutput(0);

        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([
                tok1,
                tok2 = bb.RetainOutput(1),
                y1_value,
                y2_value,
                &backprop_id1,
                &backprop_id2,
                &y1_body,
                &y2_body,
                double_backprop_opt
            ](BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be still alive.
                ASSERT_NE(nullptr, y1_body.lock());
                ASSERT_NE(nullptr, y2_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);
                const Array& y2 = bctx.GetRetainedOutput(tok2);

                testing::ExpectEqual(y1_value, y1);
                testing::ExpectEqual(y2_value, y2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                    EXPECT_TRUE(y2.IsGradRequired(backprop_id1));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id1));
                    EXPECT_FALSE(y2.IsGradRequired(backprop_id1));
                }
                EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                EXPECT_FALSE(y2.IsGradRequired(backprop_id2));

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                const Array& y2_again = bctx.GetRetainedOutput(tok2);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));
                EXPECT_EQ(internal::GetArrayBody(y2_again), internal::GetArrayBody(y2));

                Array gy1gx1 = bctx.output_grad(0) * (3 * y1 - y2) / 2;
                Array gy2gx1 = bctx.output_grad(1) * (-y1 + 3 * y2) / 2;
                bctx.input_grad() = gy1gx1 + gy2gx1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            assert(bt);
            bt.Define([
                tok1,
                tok2 = bb.RetainOutput(1),
                y1_value,
                y2_value,
                &backprop_id1,
                &backprop_id2,
                &y1_body,
                &y2_body,
                double_backprop_opt
            ](BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be still alive.
                ASSERT_NE(nullptr, y1_body.lock());
                ASSERT_NE(nullptr, y2_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);
                const Array& y2 = bctx.GetRetainedOutput(tok2);

                testing::ExpectEqual(y1_value, y1);
                testing::ExpectEqual(y2_value, y2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                    EXPECT_TRUE(y2.IsGradRequired(backprop_id1));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id1));
                    EXPECT_FALSE(y2.IsGradRequired(backprop_id1));
                }
                EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                EXPECT_FALSE(y2.IsGradRequired(backprop_id2));

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                const Array& y2_again = bctx.GetRetainedOutput(tok2);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));
                EXPECT_EQ(internal::GetArrayBody(y2_again), internal::GetArrayBody(y2));

                Array gy1gx2 = bctx.output_grad(0) * (3 * y1 + y2) / 2;
                Array gy2gx2 = bctx.output_grad(1) * (y1 + 3 * y2) / 2;
                bctx.input_grad() = gy1gx2 + gy2gx2;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array expected_x1_grad = 2 * Exp(x1_value + 2 * x2_value);
        Array expected_x2_grad = 4 * Exp(x1_value + 2 * x2_value);
        Array y1{};
        Array y2{};
        {
            forward(x1, x2, y1, y2);

            // Keep weak references to y's to check if they are actually kept alive.
            y1_body = internal::GetArrayBody(y1);
            y2_body = internal::GetArrayBody(y2);
        }
        // y's are alive here
        Backward({y1, y2}, backprop_id1, double_backprop_opt);
        testing::ExpectAllClose(expected_x1_grad, *x1.GetGrad(backprop_id1));
        testing::ExpectAllClose(expected_x2_grad, *x2.GetGrad(backprop_id1));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainOutputTest, RetainOutput_FallBackToPreviousArrayNode) {
    // This test checks retained output arrays can be retrieved, where their array bodyies (yn) is gone at the moment of retrieval, but
    // their array nodes are still alive.
    //
    // (x1) <- [forward] <- (y1 := exp(x1 + 2 x2) + exp(2 x1 + x2)) <- [view] <- (z1)
    // (x2) <-           <- (y2 := exp(x1 + 2 x2) - exp(2 x1 + x2)) <- [view] <- (z2)
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    std::weak_ptr<internal::ArrayBody> y1_body{};
    std::weak_ptr<internal::ArrayBody> y2_body{};

    auto forward = [&backprop_id1, &backprop_id2, &y1_body, &y2_body, double_backprop_opt](
                           const Array& x1, const Array& x2, Array& y1, Array& y2) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = Exp(x1_c + 2 * x2_c) + Exp(2 * x1_c + x2_c);
        y2 = Exp(x1_c + 2 * x2_c) - Exp(2 * x1_c + x2_c);

        Array y1_value = y1.MakeView();
        Array y2_value = y2.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, {y1, y2}};

        // y1 is retrieved with copied tokens.
        // y2 is retrieved with sperarately acquired tokens.
        RetainedOutputToken tok1 = bb.RetainOutput(0);

        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([
                tok1,
                tok2 = bb.RetainOutput(1),
                y1_value,
                y2_value,
                &backprop_id1,
                &backprop_id2,
                &y1_body,
                &y2_body,
                double_backprop_opt
            ](BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());
                ASSERT_EQ(nullptr, y2_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);
                const Array& y2 = bctx.GetRetainedOutput(tok2);

                testing::ExpectEqual(y1_value, y1);
                testing::ExpectEqual(y2_value, y2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                    EXPECT_TRUE(y2.IsGradRequired(backprop_id1));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id1));
                    EXPECT_FALSE(y2.IsGradRequired(backprop_id1));
                }
                EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                EXPECT_FALSE(y2.IsGradRequired(backprop_id2));

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                const Array& y2_again = bctx.GetRetainedOutput(tok2);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));
                EXPECT_EQ(internal::GetArrayBody(y2_again), internal::GetArrayBody(y2));

                Array gy1gx1 = bctx.output_grad(0) * (3 * y1 - y2) / 2;
                Array gy2gx1 = bctx.output_grad(1) * (-y1 + 3 * y2) / 2;
                bctx.input_grad() = gy1gx1 + gy2gx1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([
                tok1,
                tok2 = bb.RetainOutput(1),
                y1_value,
                y2_value,
                &backprop_id1,
                &backprop_id2,
                &y1_body,
                &y2_body,
                double_backprop_opt
            ](BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());
                ASSERT_EQ(nullptr, y2_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);
                const Array& y2 = bctx.GetRetainedOutput(tok2);

                testing::ExpectEqual(y1_value, y1);
                testing::ExpectEqual(y2_value, y2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                    EXPECT_TRUE(y2.IsGradRequired(backprop_id1));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id1));
                    EXPECT_FALSE(y2.IsGradRequired(backprop_id1));
                }
                EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                EXPECT_FALSE(y2.IsGradRequired(backprop_id2));

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                const Array& y2_again = bctx.GetRetainedOutput(tok2);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));
                EXPECT_EQ(internal::GetArrayBody(y2_again), internal::GetArrayBody(y2));

                Array gy1gx2 = bctx.output_grad(0) * (3 * y1 + y2) / 2;
                Array gy2gx2 = bctx.output_grad(1) * (y1 + 3 * y2) / 2;
                bctx.input_grad() = gy1gx2 + gy2gx2;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array expected_x1_grad = 2 * Exp(x1_value + 2 * x2_value);
        Array expected_x2_grad = 4 * Exp(x1_value + 2 * x2_value);
        Array z1{};
        Array z2{};
        {
            Array y1{};
            Array y2{};
            forward(x1, x2, y1, y2);

            // Keep weak references to y's to check if they are actually dead.
            y1_body = internal::GetArrayBody(y1);
            y2_body = internal::GetArrayBody(y2);
            z1 = y1.MakeView();
            z2 = y2.MakeView();
        }
        // y's are dead here
        Backward({z1, z2}, backprop_id1, double_backprop_opt);
        testing::ExpectAllClose(expected_x1_grad, *x1.GetGrad(backprop_id1));
        testing::ExpectAllClose(expected_x2_grad, *x2.GetGrad(backprop_id1));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainOutputTest, RetainOutput_PreviousArrayNodeOfBackwardGraphIsDead) {
    // This test checks retained output arrays can be retrieved, where their array bodies (yn) are gone at the moment of retrieval, and
    // their array nodes are also gone. In that case new mocked previous array nodes are created.
    //
    // (x1) <- [forward] <- (y1 := exp(x1 + 2 x2) + exp(2 x1 + x2))
    // (x2) <-           <- (y2 := exp(x1 + 2 x2) - exp(2 x1 + x2)) <- [view] <- (z2)
    //
    // In the backward of y2, the value of y1 would be wanted but it's gone.
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    std::weak_ptr<internal::ArrayBody> y1_body{};
    std::weak_ptr<const internal::ArrayNode> y1_node{};

    auto forward = [&backprop_id1, &backprop_id2, &y1_body, &y1_node, double_backprop_opt](
                           const Array& x1, const Array& x2, Array& y1, Array& y2) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = Exp(x1_c + 2 * x2_c) + Exp(2 * x1_c + x2_c);
        y2 = Exp(x1_c + 2 * x2_c) - Exp(2 * x1_c + x2_c);

        Array y1_value = y1.MakeView();
        Array y2_value = y2.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, {y1, y2}};

        // y1 is retrieved with copied tokens.
        // y2 is retrieved with sperarately acquired tokens.
        RetainedOutputToken tok1 = bb.RetainOutput(0);

        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([
                tok1,
                tok2 = bb.RetainOutput(1),
                y1_value,
                y2_value,
                &backprop_id1,
                &backprop_id2,
                &y1_body,
                &y1_node,
                double_backprop_opt
            ](BackwardContext & bctx) {
                // Test assumption: the body and node of y1 must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());
                ASSERT_EQ(nullptr, y1_node.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);
                const Array& y2 = bctx.GetRetainedOutput(tok2);

                testing::ExpectEqual(y1_value, y1);
                testing::ExpectEqual(y2_value, y2);
                EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(y2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id2));
                    EXPECT_TRUE(y2.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                    EXPECT_FALSE(y2.IsGradRequired(backprop_id2));
                }

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                const Array& y2_again = bctx.GetRetainedOutput(tok2);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));
                EXPECT_EQ(internal::GetArrayBody(y2_again), internal::GetArrayBody(y2));

                Array gy1gx1 = bctx.output_grad(0) * (3 * y1 - y2) / 2;
                Array gy2gx1 = bctx.output_grad(1) * (-y1 + 3 * y2) / 2;
                bctx.input_grad() = gy1gx1 + gy2gx1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([
                tok1,
                tok2 = bb.RetainOutput(1),
                y1_value,
                y2_value,
                &backprop_id1,
                &backprop_id2,
                &y1_body,
                &y1_node,
                double_backprop_opt
            ](BackwardContext & bctx) {
                // Test assumption: the body and node of y1 must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());
                ASSERT_EQ(nullptr, y1_node.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);
                const Array& y2 = bctx.GetRetainedOutput(tok2);

                testing::ExpectEqual(y1_value, y1);
                testing::ExpectEqual(y2_value, y2);
                EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(y2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id2));
                    EXPECT_TRUE(y2.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                    EXPECT_FALSE(y2.IsGradRequired(backprop_id2));
                }

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                const Array& y2_again = bctx.GetRetainedOutput(tok2);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));
                EXPECT_EQ(internal::GetArrayBody(y2_again), internal::GetArrayBody(y2));

                Array gy1gx2 = bctx.output_grad(0) * (3 * y1 + y2) / 2;
                Array gy2gx2 = bctx.output_grad(1) * (y1 + 3 * y2) / 2;
                bctx.input_grad() = gy1gx2 + gy2gx2;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array expected_x1_grad = 1 * Exp(x1_value + 2 * x2_value) - 2 * Exp(2 * x1_value + x2_value);
        Array expected_x2_grad = 2 * Exp(x1_value + 2 * x2_value) - 1 * Exp(2 * x1_value + x2_value);
        Array z2{};
        {
            Array y1{};
            Array y2{};
            forward(x1, x2, y1, y2);

            // Keep weak reference to y1.body() and y1's node to check if it is actually gone
            y1_body = internal::GetArrayBody(y1);
            y1_node = internal::GetArrayBody(y1)->GetArrayNode(backprop_id2);
            // Only z2 is kept. y1 (and therefore y1's node) will be released.
            z2 = y2.MakeView();
        }
        // Only z2 is alive here
        Backward({z2}, backprop_id2, double_backprop_opt);
        testing::ExpectAllClose(expected_x1_grad, *x1.GetGrad(backprop_id2));
        testing::ExpectAllClose(expected_x2_grad, *x2.GetGrad(backprop_id2));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainOutputTest, RetainOutput_NonOverlappingGraphsInInputArrays) {
    // This test checks retained output arrays can be retrieved and belongs to the correct graphs, even when input arrays have disjoint set
    // of graphs w.r.t. backward definitions.
    //
    // (x1) <- [forward] <- (y1 := exp(2 x1 + x2)) <- [view] <- (z1)
    // (x2) <-
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    std::weak_ptr<internal::ArrayBody> y1_body{};

    auto forward = [&backprop_id1, &backprop_id2, &y1_body, double_backprop_opt](const Array& x1, const Array& x2, Array& y1) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = Exp(2 * x1_c + x2_c);

        Array y1_value = y1.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, y1};
        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            assert(bt);
            bt.Define([ tok1 = bb.RetainOutput(0), y1_value, &backprop_id1, &backprop_id2, &y1_body, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);

                testing::ExpectEqual(y1_value, y1);
                EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                }

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));

                bctx.input_grad() = bctx.output_grad(0) * y1 * 2;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            assert(bt);
            bt.Define([ tok1 = bb.RetainOutput(0), y1_value, &backprop_id1, &backprop_id2, &y1_body, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(tok1);

                testing::ExpectEqual(y1_value, y1);
                EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id2));
                }

                // Retrieve retained outputs repeatedly
                const Array& y1_again = bctx.GetRetainedOutput(tok1);
                EXPECT_EQ(internal::GetArrayBody(y1_again), internal::GetArrayBody(y1));

                bctx.input_grad() = bctx.output_grad(0) * y1;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id2);
        Array expected_x2_grad = 1 * Exp(2 * x1_value + x2_value);
        Array z1{};
        {
            Array y1{};
            forward(x1, x2, y1);

            // Keep weak references to y's to check if they are actually dead.
            y1_body = internal::GetArrayBody(y1);
            z1 = y1.MakeView();
        }
        // y's are dead here
        Backward({z1}, backprop_id2, double_backprop_opt);
        testing::ExpectAllClose(expected_x2_grad, *x2.GetGrad(backprop_id2));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainOutputTest, RetainOutput_NonOverlappingGraphsInInputArraysManyInputs) {
    // This test checks that retained output arrays can be retrieved when some of the inputs does not belong to a certain graph.
    //
    // (x1) <- [forward] <- (y1 := Exp(x1 + x2 + x3)) <- [view] <- (z1)
    // (x2) <-
    // (x3) <-
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropScope backprop_scope3{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();
    BackpropId backprop_id3 = backprop_scope3.backprop_id();

    std::weak_ptr<internal::ArrayBody> y1_body{};

    auto forward = [&backprop_id1, &backprop_id2, &backprop_id3, &y1_body, double_backprop_opt](
                           const Array& x1, const Array& x2, const Array& x3, Array& y1) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        Array x3_c = x3.AsGradStopped();
        y1 = Exp(x1_c + x2_c + x3_c);

        Array y1_value = y1.MakeView();

        BackwardBuilder bb{"func", {x1, x2, x3}, y1};
        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([](BackwardContext& /*bctx*/) { FAIL() << "This code should not be executed in this test"; });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([](BackwardContext& /*bctx*/) { FAIL() << "This code should not be executed in this test"; });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(2);
            bt.Define([ y1_tok = bb.RetainOutput(0), y1_value, &backprop_id1, &backprop_id2, &backprop_id3, &y1_body, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(y1_tok);

                testing::ExpectEqual(y1_value, y1);
                EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(y1.IsGradRequired(backprop_id2));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id3));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id3));
                }

                bctx.input_grad() = bctx.output_grad(0) * y1;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x3_value = testing::BuildArray({1}).WithLinearData<double>(4);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id2);
        Array x3 = x3_value.MakeView().RequireGrad(backprop_id3);
        Array expected_x3_grad = Exp(x1_value + x2_value + x3_value);
        Array z1{};
        {
            Array y1{};
            forward(x1, x2, x3, y1);

            // Keep weak references to y's to check if they are actually dead.
            y1_body = internal::GetArrayBody(y1);
            z1 = y1.MakeView();
        }
        // y's are dead here
        Backward({z1}, backprop_id3, double_backprop_opt);
        testing::ExpectAllClose(expected_x3_grad, *x3.GetGrad(backprop_id3));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainOutputTest, RetainOutput_NonOverlappingGraphsInInputArraysManyInputsReversedDefineOrder) {
    // This test checks that retained output arrays can be retrieved when some of the inputs does not belong to a certain graph.
    // Even when the order of backward definitions are not in the order the inputs are given to the builder.
    //
    // (x1) <- [forward] <- (y1 := Exp(x1 + x2 + x3)) <- [view] <- (z1)
    // (x2) <-
    // (x3) <-
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropScope backprop_scope3{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();
    BackpropId backprop_id3 = backprop_scope3.backprop_id();

    std::weak_ptr<internal::ArrayBody> y1_body{};

    auto forward = [&backprop_id1, &backprop_id2, &backprop_id3, &y1_body, double_backprop_opt](
                           const Array& x1, const Array& x2, const Array& x3, Array& y1) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        Array x3_c = x3.AsGradStopped();
        y1 = Exp(x1_c + x2_c + x3_c);

        Array y1_value = y1.MakeView();

        BackwardBuilder bb{"func", {x1, x2, x3}, y1};
        {
            BackwardBuilder::Target bt = bb.CreateTarget(2);
            bt.Define([ y1_tok = bb.RetainOutput(0), y1_value, &backprop_id1, &backprop_id2, &backprop_id3, &y1_body, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Test assumption: the bodies of ys must be dead.
                ASSERT_EQ(nullptr, y1_body.lock());

                // Retrieve retained outputs
                const Array& y1 = bctx.GetRetainedOutput(y1_tok);

                testing::ExpectEqual(y1_value, y1);
                EXPECT_TRUE(y1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(y1.IsGradRequired(backprop_id2));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(y1.IsGradRequired(backprop_id3));
                } else {
                    EXPECT_FALSE(y1.IsGradRequired(backprop_id3));
                }

                bctx.input_grad() = bctx.output_grad(0) * y1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([](BackwardContext& /*bctx*/) { FAIL() << "This code should not be executed in this test"; });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([](BackwardContext& /*bctx*/) { FAIL() << "This code should not be executed in this test"; });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x3_value = testing::BuildArray({1}).WithLinearData<double>(4);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id2);
        Array x3 = x3_value.MakeView().RequireGrad(backprop_id3);
        Array expected_x3_grad = Exp(x1_value + x2_value + x3_value);
        Array z1{};
        {
            Array y1{};
            forward(x1, x2, x3, y1);

            // Keep weak references to y's to check if they are actually dead.
            y1_body = internal::GetArrayBody(y1);
            z1 = y1.MakeView();
        }
        // y's are dead here
        Backward({z1}, backprop_id3, double_backprop_opt);
        testing::ExpectAllClose(expected_x3_grad, *x3.GetGrad(backprop_id3));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

INSTANTIATE_TEST_CASE_P(Params, BackpropRetainOutputTest, ::testing::Values(DoubleBackpropOption::kDisable, DoubleBackpropOption::kEnable));

class BackpropRetainInputTest : public ::testing::TestWithParam<DoubleBackpropOption> {};

TEST_P(BackpropRetainInputTest, RetainInput) {
    // (x1) <- [forward] <- (y1 := x1 ^ 2 * x2 ^ 2)
    // (x2) <-           <- (y2 := x1 ^ 3 * x2 ^ 3)
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    auto forward = [&backprop_id1, &backprop_id2, double_backprop_opt](const Array& x1, const Array& x2, Array& y1, Array& y2) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = x1_c * x1_c * x2_c * x2_c;
        y2 = x1_c * x1_c * x1_c * x2_c * x2_c * x2_c;

        Array y1_value = y1.MakeView();
        Array y2_value = y2.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, {y1, y2}};

        // x1 is retrieved with copied tokens.
        // x2 is retrieved with sperarately acquired tokens.
        RetainedInputToken tok1 = bb.RetainInput(0);

        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([ tok1, tok2 = bb.RetainInput(1), x1_c, x2_c, &backprop_id1, &backprop_id2, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Retrieve retained inputs
                const Array& x1 = bctx.GetRetainedInput(tok1);
                const Array& x2 = bctx.GetRetainedInput(tok2);

                testing::ExpectEqual(x1_c, x1);
                testing::ExpectEqual(x2_c, x2);
                EXPECT_TRUE(x1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(x2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(x1.IsGradRequired(backprop_id2));
                    EXPECT_TRUE(x2.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(x1.IsGradRequired(backprop_id2));
                    EXPECT_FALSE(x2.IsGradRequired(backprop_id2));
                }

                // Retrieve retained inputs repeatedly
                const Array& x1_again = bctx.GetRetainedInput(tok1);
                const Array& x2_again = bctx.GetRetainedInput(tok2);
                EXPECT_EQ(internal::GetArrayBody(x1_again), internal::GetArrayBody(x1));
                EXPECT_EQ(internal::GetArrayBody(x2_again), internal::GetArrayBody(x2));

                Array gy1gx1 = bctx.output_grad(0) * 2 * x1 * x2 * x2;
                Array gy2gx1 = bctx.output_grad(1) * 3 * x1 * x1 * x2 * x2 * x2;
                bctx.input_grad() = gy1gx1 + gy2gx1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([ tok1, tok2 = bb.RetainInput(1), x1_c, x2_c, &backprop_id1, &backprop_id2, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Retrieve retained outputs
                const Array& x1 = bctx.GetRetainedInput(tok1);
                const Array& x2 = bctx.GetRetainedInput(tok2);

                testing::ExpectEqual(x1_c, x1);
                testing::ExpectEqual(x2_c, x2);
                EXPECT_TRUE(x1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(x2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(x1.IsGradRequired(backprop_id2));
                    EXPECT_TRUE(x2.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(x1.IsGradRequired(backprop_id2));
                    EXPECT_FALSE(x2.IsGradRequired(backprop_id2));
                }

                // Retrieve retained outputs repeatedly
                const Array& x1_again = bctx.GetRetainedInput(tok1);
                const Array& x2_again = bctx.GetRetainedInput(tok2);
                EXPECT_EQ(internal::GetArrayBody(x1_again), internal::GetArrayBody(x1));
                EXPECT_EQ(internal::GetArrayBody(x2_again), internal::GetArrayBody(x2));

                Array gy1gx2 = bctx.output_grad(0) * x1 * x1 * 2 * x2;
                Array gy2gx2 = bctx.output_grad(1) * x1 * x1 * x1 * 3 * x2 * x2;
                bctx.input_grad() = gy1gx2 + gy2gx2;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array expected_x1_grad = 2 * x1_value * x2_value * x2_value + 3 * x1_value * x1_value * x2_value * x2_value * x2_value;
        Array expected_x2_grad = x1_value * x1_value * 2 * x2_value + x1_value * x1_value * x1_value * 3 * x2_value * x2_value;

        Array y1{};
        Array y2{};
        forward(x1, x2, y1, y2);

        Backward({y1, y2}, backprop_id2, double_backprop_opt);
        testing::ExpectEqual(expected_x1_grad, *x1.GetGrad(backprop_id2));
        testing::ExpectEqual(expected_x2_grad, *x2.GetGrad(backprop_id2));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainInputTest, RetainInputArrayBodyIsDead) {
    // This test checks that input array retention works when the input array body is dead.
    //
    // (x1) <- [forward] <- (y1 := x1 ^ 2 * x2 ^ 2)
    // (x2) <-           <- (y2 := x1 ^ 3 * x2 ^ 3)
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    std::weak_ptr<internal::ArrayBody> x1_body{};
    std::weak_ptr<internal::ArrayBody> x2_body{};

    auto forward = [&backprop_id1, &backprop_id2, &x1_body, &x2_body, double_backprop_opt](
                           const Array& x1, const Array& x2, Array& y1, Array& y2) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = x1_c * x1_c * x2_c * x2_c;
        y2 = x1_c * x1_c * x1_c * x2_c * x2_c * x2_c;

        Array y1_value = y1.MakeView();
        Array y2_value = y2.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, {y1, y2}};

        // x1 is retrieved with copied tokens.
        // x2 is retrieved with sperarately acquired tokens.
        RetainedInputToken tok1 = bb.RetainInput(0);

        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([ tok1, tok2 = bb.RetainInput(1), x1_c, x2_c, &backprop_id1, &backprop_id2, &x1_body, &x2_body, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Test assumption: the array bodies must be gone.
                EXPECT_EQ(x1_body.lock(), nullptr);
                EXPECT_EQ(x2_body.lock(), nullptr);

                // Retrieve retained inputs
                const Array& x1 = bctx.GetRetainedInput(tok1);
                const Array& x2 = bctx.GetRetainedInput(tok2);

                testing::ExpectEqual(x1_c, x1);
                testing::ExpectEqual(x2_c, x2);
                EXPECT_TRUE(x1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(x2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(x1.IsGradRequired(backprop_id2));
                    EXPECT_TRUE(x2.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(x1.IsGradRequired(backprop_id2));
                    EXPECT_FALSE(x2.IsGradRequired(backprop_id2));
                }

                // Retrieve retained inputs repeatedly
                const Array& x1_again = bctx.GetRetainedInput(tok1);
                const Array& x2_again = bctx.GetRetainedInput(tok2);
                EXPECT_EQ(internal::GetArrayBody(x1_again), internal::GetArrayBody(x1));
                EXPECT_EQ(internal::GetArrayBody(x2_again), internal::GetArrayBody(x2));

                Array gy1gx1 = bctx.output_grad(0) * 2 * x1 * x2 * x2;
                Array gy2gx1 = bctx.output_grad(1) * 3 * x1 * x1 * x2 * x2 * x2;
                bctx.input_grad() = gy1gx1 + gy2gx1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([ tok1, tok2 = bb.RetainInput(1), x1_c, x2_c, &backprop_id1, &backprop_id2, &x1_body, &x2_body, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Test assumption: the array bodies must be gone.
                EXPECT_EQ(x1_body.lock(), nullptr);
                EXPECT_EQ(x2_body.lock(), nullptr);

                // Retrieve retained outputs
                const Array& x1 = bctx.GetRetainedInput(tok1);
                const Array& x2 = bctx.GetRetainedInput(tok2);

                testing::ExpectEqual(x1_c, x1);
                testing::ExpectEqual(x2_c, x2);
                EXPECT_TRUE(x1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(x2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(x1.IsGradRequired(backprop_id2));
                    EXPECT_TRUE(x2.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(x1.IsGradRequired(backprop_id2));
                    EXPECT_FALSE(x2.IsGradRequired(backprop_id2));
                }

                // Retrieve retained outputs repeatedly
                const Array& x1_again = bctx.GetRetainedInput(tok1);
                const Array& x2_again = bctx.GetRetainedInput(tok2);
                EXPECT_EQ(internal::GetArrayBody(x1_again), internal::GetArrayBody(x1));
                EXPECT_EQ(internal::GetArrayBody(x2_again), internal::GetArrayBody(x2));

                Array gy1gx2 = bctx.output_grad(0) * x1 * x1 * 2 * x2;
                Array gy2gx2 = bctx.output_grad(1) * x1 * x1 * x1 * 3 * x2 * x2;
                bctx.input_grad() = gy1gx2 + gy2gx2;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);

        Array y1{};
        Array y2{};

        {
            Array x1 = x1_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
            Array x2 = x2_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);

            forward(x1, x2, y1, y2);

            x1_body = internal::GetArrayBody(x1);
            x2_body = internal::GetArrayBody(x2);
        }

        Backward({y1, y2}, backprop_id2, double_backprop_opt);
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

TEST_P(BackpropRetainInputTest, RetainInputWithDifferentGraphs) {
    // (x1) <- [forward] <- (y1 := x1 ^ 2 * x2 ^ 2)
    // (x2) <-           <- (y2 := x1 ^ 3 * x2 ^ 3)
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    DoubleBackpropOption double_backprop_opt = GetParam();
    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    auto forward = [&backprop_id1, &backprop_id2, double_backprop_opt](const Array& x1, const Array& x2, Array& y1, Array& y2) {
        Array x1_c = x1.AsGradStopped();
        Array x2_c = x2.AsGradStopped();
        y1 = x1_c * x1_c * x2_c * x2_c;
        y2 = x1_c * x1_c * x1_c * x2_c * x2_c * x2_c;

        Array y1_value = y1.MakeView();
        Array y2_value = y2.MakeView();

        BackwardBuilder bb{"func", {x1, x2}, {y1, y2}};

        // x1 is retrieved with copied tokens.
        // x2 is retrieved with sperarately acquired tokens.
        RetainedInputToken tok1 = bb.RetainInput(0);

        {
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([ tok1, tok2 = bb.RetainInput(1), x1_c, x2_c, &backprop_id1, &backprop_id2, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Retrieve retained inputs
                const Array& x1 = bctx.GetRetainedInput(tok1);
                const Array& x2 = bctx.GetRetainedInput(tok2);

                testing::ExpectEqual(x1_c, x1);
                testing::ExpectEqual(x2_c, x2);
                EXPECT_TRUE(x1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(x2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(x1.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(x1.IsGradRequired(backprop_id2));
                }
                EXPECT_FALSE(x2.IsGradRequired(backprop_id2));

                // Retrieve retained inputs repeatedly
                const Array& x1_again = bctx.GetRetainedInput(tok1);
                const Array& x2_again = bctx.GetRetainedInput(tok2);
                EXPECT_EQ(internal::GetArrayBody(x1_again), internal::GetArrayBody(x1));
                EXPECT_EQ(internal::GetArrayBody(x2_again), internal::GetArrayBody(x2));

                Array gy1gx1 = bctx.output_grad(0) * 2 * x1 * x2 * x2;
                Array gy2gx1 = bctx.output_grad(1) * 3 * x1 * x1 * x2 * x2 * x2;
                bctx.input_grad() = gy1gx1 + gy2gx1;
            });
        }
        {
            BackwardBuilder::Target bt = bb.CreateTarget(1);
            bt.Define([ tok1, tok2 = bb.RetainInput(1), x1_c, x2_c, &backprop_id1, &backprop_id2, double_backprop_opt ](
                    BackwardContext & bctx) {
                // Retrieve retained outputs
                const Array& x1 = bctx.GetRetainedInput(tok1);
                const Array& x2 = bctx.GetRetainedInput(tok2);

                testing::ExpectEqual(x1_c, x1);
                testing::ExpectEqual(x2_c, x2);
                EXPECT_TRUE(x1.IsGradRequired(backprop_id1));
                EXPECT_TRUE(x2.IsGradRequired(backprop_id1));
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(x1.IsGradRequired(backprop_id2));
                } else {
                    EXPECT_FALSE(x1.IsGradRequired(backprop_id2));
                }
                EXPECT_FALSE(x2.IsGradRequired(backprop_id2));

                // Retrieve retained outputs repeatedly
                const Array& x1_again = bctx.GetRetainedInput(tok1);
                const Array& x2_again = bctx.GetRetainedInput(tok2);
                EXPECT_EQ(internal::GetArrayBody(x1_again), internal::GetArrayBody(x1));
                EXPECT_EQ(internal::GetArrayBody(x2_again), internal::GetArrayBody(x2));

                Array gy1gx2 = bctx.output_grad(0) * x1 * x1 * 2 * x2;
                Array gy2gx2 = bctx.output_grad(1) * x1 * x1 * x1 * 3 * x2 * x2;
                bctx.input_grad() = gy1gx2 + gy2gx2;
            });
        }
    };

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};

        Array x1_value = testing::BuildArray({1}).WithLinearData<double>(2);
        Array x2_value = testing::BuildArray({1}).WithLinearData<double>(3);
        Array x1 = x1_value.MakeView().RequireGrad(backprop_id1).RequireGrad(backprop_id2);
        Array x2 = x2_value.MakeView().RequireGrad(backprop_id1);
        Array expected_x1_grad = 2 * x1_value * x2_value * x2_value + 3 * x1_value * x1_value * x2_value * x2_value * x2_value;

        Array y1{};
        Array y2{};
        forward(x1, x2, y1, y2);

        Backward({y1, y2}, backprop_id2, double_backprop_opt);
        testing::ExpectEqual(expected_x1_grad, *x1.GetGrad(backprop_id2));
    }
    EXPECT_TRUE(IsAllArrayBodiesFreed(tracker));
}

INSTANTIATE_TEST_CASE_P(Params, BackpropRetainInputTest, ::testing::Values(DoubleBackpropOption::kDisable, DoubleBackpropOption::kEnable));

TEST(BackpropGradValidationTest, InvalidGradShape) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});

    auto forward = [](const Array& x1, Array& y1) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", x1, y1};
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);
                EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                bctx.input_grad() = gy1.Reshape({2, 1});  // Intentionally set to a wrong shape (2, 1), instead of (2,).
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    forward(x1, y1);

    y1.SetGrad(gy1_value, backprop_id);

    // The shape of the computed gradient of x1 is (2, 1) but the shape of x1 is (2,), thus an exception should be thrown.
    EXPECT_THROW(Backward({y1}, backprop_id, DoubleBackpropOption::kDisable), DimensionError);
}

TEST(BackpropGradValidationTest, InvalidGradDtype) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});

    auto forward = [](const Array& x1, Array& y1) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", x1, y1};
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);
                EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                bctx.input_grad() = gy1.AsType(Dtype::kFloat32);  // Intentionally set to a wrong dtype float, instead of double.
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    forward(x1, y1);

    y1.SetGrad(gy1_value, backprop_id);

    // The dtype of the computed gradient of x1 is float but the dtype of x1 is double, thus an exception should be thrown.
    EXPECT_THROW(Backward({y1}, backprop_id, DoubleBackpropOption::kDisable), DtypeError);
}

TEST(BackpropGradValidationTest, InvalidGradDevice) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    BackpropScope backprop_scope{"bp"};
    BackpropId backprop_id = backprop_scope.backprop_id();
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});

    auto forward = [](const Array& x1, Array& y1) {
        ASSERT_TRUE(x1.IsGradRequired(AnyGraph{}));
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_FALSE(y1.IsGradRequired(AnyGraph{}));

        {
            BackwardBuilder bb{"func", x1, y1};
            BackwardBuilder::Target bt = bb.CreateTarget(0);
            bt.Define([& device = x1.device()](BackwardContext & bctx) {
                const Array& gy1 = bctx.output_grad(0);
                EXPECT_FALSE(gy1.IsGradRequired(AnyGraph{}));
                bctx.input_grad() =
                        gy1.ToDevice(device.backend().GetDevice(device.index() + 1));  // Intentionally set to a different device.
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(backprop_id);
    Array y1{};
    forward(x1, y1);

    y1.SetGrad(gy1_value, backprop_id);

    // The device of the computed gradient of x1 is on a different device from the device of x1, thus an exception should be throws.
    EXPECT_THROW(Backward({y1}, backprop_id, DoubleBackpropOption::kDisable), DeviceError);
}

}  // namespace
}  // namespace xchainer
