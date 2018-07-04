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
#include "xchainer/array_node.h"
#include "xchainer/backend.h"
#include "xchainer/check_backward.h"
#include "xchainer/context.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
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
        Backward(z, kDefaultGraphId, DoubleBackpropOption::kEnable);
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

TEST_P(BackpropTest, MultipleGraphsDoubleBackprop) {
    GraphId graph_x = "graph_x";
    GraphId graph_y = "graph_y";

    auto x = Full({1}, 2.0f);
    x.RequireGrad(graph_x);

    auto y = Full({1}, 3.0f);
    y.RequireGrad(graph_y);

    auto z = x * (x + y);
    Backward(z, graph_x);

    auto gx = *x.GetGrad(graph_x);  // 2x + y
    EXPECT_FALSE(gx.IsGradRequired(graph_x));
    EXPECT_TRUE(gx.IsGradRequired(graph_y));

    auto w = x * gx;
    Backward(w, graph_y);

    auto e = Full({1}, 2.0f);
    ExpectEqual<float>(e, *y.GetGrad(graph_y));  // x
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

    GraphId graph_id_1 = "graph_1";
    GraphId graph_id_2 = "graph_2";

    x1.RequireGrad(graph_id_1);
    x2.RequireGrad(graph_id_2);

    Array y1 = x1 * x2;
    Backward(y1, graph_id_1);

    Array expected_1 = MakeFullArray({1}, {5.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(graph_id_1));
    EXPECT_FALSE(x2.GetGrad(graph_id_2));
}

TEST_P(BackpropTest, MultipleGraphsSameInput) {
    Array x1 = MakeFullArray({1}, {3.0f});

    GraphId graph_id_1 = "graph_1";

    x1.RequireGrad(graph_id_1);

    Array y1 = x1 * x1;
    Backward(y1, graph_id_1);

    Array expected_1 = MakeFullArray({1}, {6.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(graph_id_1));

    EXPECT_FALSE(x1.GetGrad(graph_id_1)->IsGradRequired(graph_id_1));
}

TEST_P(BackpropTest, MultipleGraphsNonExisting) {
    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    GraphId graph_id_1 = "graph_1";
    GraphId graph_id_2 = "graph_2";

    x1.RequireGrad(graph_id_1);
    x2.RequireGrad(graph_id_1);

    Array y1 = x1 * x2;
    EXPECT_THROW(Backward(y1, graph_id_2), XchainerError);
}

TEST_P(BackpropTest, MultipleGraphsReuse) {
    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    GraphId graph_id_1 = "graph_1";
    GraphId graph_id_2 = "graph_2";

    x1.RequireGrad(graph_id_1);
    x2.RequireGrad(graph_id_2);

    Array y1 = x1 * x2;
    Backward(y1, graph_id_1);

    Array expected_1 = MakeFullArray({1}, {5.0f});
    ExpectEqual<float>(expected_1, *x1.GetGrad(graph_id_1));
    EXPECT_FALSE(x2.GetGrad(graph_id_2));

    x1.ClearGrad(graph_id_1);
    x2.ClearGrad(graph_id_2);

    Array y2 = x1 * x2;
    Backward(y2, graph_id_2);

    Array expected_2 = MakeFullArray({1}, {2.0f});
    ExpectEqual<float>(expected_2, *x2.GetGrad(graph_id_2));
    EXPECT_FALSE(x1.GetGrad(graph_id_1));

    x1.ClearGrad(graph_id_1);
    x2.ClearGrad(graph_id_2);

    x1.RequireGrad(graph_id_2);
    x2.RequireGrad(graph_id_1);

    Array y3 = x1 * x2;
    Backward(y3, graph_id_2);

    ExpectEqual<float>(expected_1, *x1.GetGrad(graph_id_2));
    ExpectEqual<float>(expected_2, *x2.GetGrad(graph_id_2));
    EXPECT_FALSE(x1.GetGrad(graph_id_1));
    EXPECT_FALSE(x2.GetGrad(graph_id_1));
}

TEST_P(BackpropTest, NoCyclicReferenceInvolvingInputGrad) {
    // This test checks cyclic reference is not formed when the input gradient references the input array.
    // The cycle could happen if input array nodes directly owned their gradients.

    std::weak_ptr<internal::ArrayBody> x_grad_body{};

    {
        GraphId graph_id = "testgraph";

        auto forward = [](const Array& x, Array& y) {
            y = x.AsGradStopped() * x.AsGradStopped();

            BackwardBuilder bb{"func", y};
            bb.Define({x}, [x](BackwardContext& bctx) {
                // Create an input grad which references the input array.
                bctx.input_grad() = 2 * bctx.Cut(x) * bctx.output_grad();
            });
        };

        Array x = testing::BuildArray({1}).WithLinearData<float>();
        Array y{};

        x.RequireGrad(graph_id);
        forward(x, y);

        Backward(y, graph_id, DoubleBackpropOption::kEnable);

        x_grad_body = x.GetGrad(graph_id)->body();  // Keep weak pointer to the body of x.grad
    }

    // The body of x.grad must have been released.
    EXPECT_EQ(nullptr, x_grad_body.lock());
}

TEST_P(BackpropTest, SomeOfPreviousArrayNodesAreGone) {
    // This test checks the backward of a multiple-output function where one of the output arrays is gone.
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

        BackwardBuilder bb{"func", {y1, y2, y3, y4}};
        bb.Define({x}, [x](BackwardContext& bctx) {
            Array x_cut = bctx.Cut(x);
            Array gy1gx = bctx.output_grad(0) * Exp(x_cut) * x_cut;
            Array gy2gx = bctx.output_grad(1) * Exp(x_cut) * 2;
            Array gy3gx = bctx.output_grad(2) * Exp(x_cut) * 3;
            Array gy4gx = bctx.output_grad(3) * Exp(x_cut) * 4;
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
    Backward(z1, kDefaultGraphId);

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

TEST(BackpropEnableDoubleBackpropTest, Enabled) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    Array x1 = Full({2}, 1.f).RequireGrad();
    Array x2 = Full({2}, 2.f);
    Array y1 = x1 + x2;
    Array y2 = x1 * x2;
    Array z = y1 * y2;
    Backward(z, kDefaultGraphId, DoubleBackpropOption::kEnable);

    std::shared_ptr<const ArrayNode> z_array_node = internal::GetArrayNode(z);
    ASSERT_TRUE(z_array_node);

    std::shared_ptr<const OpNode> z_op_node = z_array_node->next_op_node();
    ASSERT_TRUE(z_op_node);

    auto y_array_nodes = z_op_node->next_array_nodes();
    ASSERT_EQ(2u, y_array_nodes.size());
    EXPECT_EQ(2u, z_op_node->backward_entries().size());

    for (const std::shared_ptr<ArrayNode>& y_array_node : y_array_nodes) {
        std::shared_ptr<const OpNode> y_op_node = y_array_node->next_op_node();
        ASSERT_TRUE(y_op_node);
        ASSERT_EQ(1u, y_op_node->next_array_nodes().size());
        EXPECT_EQ(1u, y_op_node->backward_entries().size());
    }
}

TEST(BackpropEnableDoubleBackpropTest, Disabled) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    Array x1 = Full({2}, 1.f).RequireGrad();
    Array x2 = Full({2}, 2.f);
    Array y1 = x1 + x2;
    Array y2 = x1 * x2;
    Array z = y1 * y2;
    std::shared_ptr<const ArrayNode> z_array_node = internal::GetArrayNode(z);
    ASSERT_TRUE(z_array_node);
    std::shared_ptr<const OpNode> z_op_node = z_array_node->next_op_node();
    ASSERT_TRUE(z_op_node);

    Backward(z);

    ASSERT_TRUE(z_array_node);
    EXPECT_FALSE(z_array_node->next_op_node());

    EXPECT_EQ(0u, z_op_node->next_array_nodes().size());
    EXPECT_EQ(0u, z_op_node->backward_entries().size());
}

class BackpropFunctionTest : public ::testing::TestWithParam<DoubleBackpropOption> {};

TEST_P(BackpropFunctionTest, OneToOneFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gx1_value = 2 * gy1_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, double_backprop_opt, &graph_id](const Array& x1, Array& y1) {
        ASSERT_FALSE(x1.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_TRUE(y1.IsConstant());

        {
            BackwardBuilder bb{"func", y1};
            bb.Define({x1}, [gy1_value, double_backprop_opt, &graph_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad();  // omit index
                testing::ExpectEqual(gy1_value, gy1);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(graph_id));
                } else {
                    EXPECT_TRUE(gy1.IsConstant());
                }
                bctx.input_grad() = 2 * gy1;  // omit index
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    forward(x1, y1);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(graph_id), graph_id);
    } else {
        y1.SetGrad(gy1_value, graph_id);
    }
    Backward({y1}, graph_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(graph_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(graph_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsGradRequired(graph_id));
    } else {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsConstant());
    }
}

TEST_P(BackpropFunctionTest, OneToMultiFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gy2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array gx1_value = 2 * gy1_value + 3 * gy2_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, gy2_value, double_backprop_opt, &graph_id](const Array& x1, Array& y1, Array& y2) {
        ASSERT_FALSE(x1.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 1;
        y2 = 3 * x1.AsGradStopped() + 2;
        ASSERT_TRUE(y1.IsConstant());
        ASSERT_TRUE(y2.IsConstant());

        {
            BackwardBuilder bb{"func", {y1, y2}};
            bb.Define({x1}, [gy1_value, gy2_value, double_backprop_opt, &graph_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);  // by index
                const Array& gy2 = bctx.output_grad(1);
                testing::ExpectEqual(gy1_value, gy1);
                testing::ExpectEqual(gy2_value, gy2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(graph_id));
                    EXPECT_TRUE(gy2.IsGradRequired(graph_id));
                } else {
                    EXPECT_TRUE(gy1.IsConstant());
                    EXPECT_TRUE(gy2.IsConstant());
                }
                bctx.input_grad(0) = 2 * gy1 + 3 * gy2;  // by index
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    Array y2{};
    forward(x1, y1, y2);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(graph_id), graph_id);
        y2.SetGrad(gy2_value.MakeView().RequireGrad(graph_id), graph_id);
    } else {
        y1.SetGrad(gy1_value, graph_id);
        y2.SetGrad(gy2_value, graph_id);
    }
    Backward({y1, y2}, graph_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(graph_id));
    testing::ExpectEqual(gy2_value, *y2.GetGrad(graph_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(graph_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsGradRequired(graph_id));
        EXPECT_TRUE(y2.GetGrad(graph_id)->IsGradRequired(graph_id));
    } else {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsConstant());
        EXPECT_TRUE(y2.GetGrad(graph_id)->IsConstant());
    }
}

TEST_P(BackpropFunctionTest, MultiToOneFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array x2_value = testing::BuildArray(shape).WithData<T>({4, -1});
    Array x3_value = testing::BuildArray(shape).WithData<T>({-1, 3});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});
    Array gx1_value = 2 * gy1_value;
    Array gx2_value = 3 * gy1_value;
    Array gx3_value = 1 * gy1_value;

    DoubleBackpropOption double_backprop_opt = GetParam();

    auto forward = [gy1_value, double_backprop_opt, &graph_id](const Array& x1, const Array& x2, const Array& x3, Array& y1) {
        ASSERT_FALSE(x1.IsConstant());
        ASSERT_FALSE(x2.IsConstant());
        ASSERT_FALSE(x3.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 3 * x2.AsGradStopped() + x3.AsGradStopped() + 1;
        ASSERT_TRUE(y1.IsConstant());

        {
            BackwardBuilder bb{"func", {y1}};
            bb.Define({x1}, [gy1_value, double_backprop_opt, &graph_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad();  // omit index
                testing::ExpectEqual(gy1_value, gy1);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(graph_id));
                } else {
                    EXPECT_TRUE(gy1.IsConstant());
                }

                // input_grad has null array
                EXPECT_EQ(nullptr, bctx.input_grad().body());

                // input_grad setter
                bctx.input_grad() = 2 * gy1;  // omit index

                // Check bctx.input_grad() as a getter
                Array gx1_back = bctx.input_grad();
                testing::ExpectEqual(2 * gy1, gx1_back);
            });
            bb.Define({x2, x3}, [gy1_value, double_backprop_opt, &graph_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);  // by index
                testing::ExpectEqual(gy1_value, gy1);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(graph_id));
                } else {
                    EXPECT_TRUE(gy1.IsConstant());
                }

                // input_grad has null array
                EXPECT_EQ(nullptr, bctx.input_grad(0).body());
                EXPECT_EQ(nullptr, bctx.input_grad(1).body());

                // input_grad setter
                bctx.input_grad(0) = 3 * gy1;  // by index
                bctx.input_grad(1) = 1 * gy1;

                // Check bctx.input_grad() as a getter
                Array gx2_back = bctx.input_grad(0);
                Array gx3_back = bctx.input_grad(1);
                testing::ExpectEqual(3 * gy1, gx2_back);
                testing::ExpectEqual(1 * gy1, gx3_back);
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array x2 = x2_value.MakeView().RequireGrad(graph_id);
    Array x3 = x3_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    forward(x1, x2, x3, y1);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(graph_id), graph_id);
    } else {
        y1.SetGrad(gy1_value, graph_id);
    }
    Backward({y1}, graph_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(graph_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(graph_id));
    testing::ExpectEqual(gx2_value, *x2.GetGrad(graph_id));
    testing::ExpectEqual(gx3_value, *x3.GetGrad(graph_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsGradRequired(graph_id));
    } else {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsConstant());
    }
}

TEST_P(BackpropFunctionTest, MultiToMultiFunc) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
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

    auto forward = [gy1_value, gy2_value, double_backprop_opt, &graph_id](
                           const Array& x1, const Array& x2, const Array& x3, Array& y1, Array& y2) {
        ASSERT_FALSE(x1.IsConstant());
        ASSERT_FALSE(x2.IsConstant());
        ASSERT_FALSE(x3.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 3 * x2.AsGradStopped() + 1;
        y2 = 3 * x1.AsGradStopped() + 1 * x2.AsGradStopped() + 2 * x3.AsGradStopped() + 4;
        ASSERT_TRUE(y1.IsConstant());
        ASSERT_TRUE(y2.IsConstant());

        {
            BackwardBuilder bb{"func", {y1, y2}};
            bb.Define({x1}, [gy1_value, gy2_value, double_backprop_opt, &graph_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);  // by index
                const Array& gy2 = bctx.output_grad(1);
                testing::ExpectEqual(gy1_value, gy1);
                testing::ExpectEqual(gy2_value, gy2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(graph_id));
                    EXPECT_TRUE(gy2.IsGradRequired(graph_id));
                } else {
                    EXPECT_TRUE(gy1.IsConstant());
                    EXPECT_TRUE(gy2.IsConstant());
                }
                bctx.input_grad(0) = 2 * gy1 + 3 * gy2;  // by index
            });
            bb.Define({x2, x3}, [gy1_value, gy2_value, double_backprop_opt, &graph_id](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);  // by index
                const Array& gy2 = bctx.output_grad(1);
                testing::ExpectEqual(gy1_value, gy1);
                testing::ExpectEqual(gy2_value, gy2);
                if (double_backprop_opt == DoubleBackpropOption::kEnable) {
                    EXPECT_TRUE(gy1.IsGradRequired(graph_id));
                    EXPECT_TRUE(gy2.IsGradRequired(graph_id));
                } else {
                    EXPECT_TRUE(gy1.IsConstant());
                    EXPECT_TRUE(gy2.IsConstant());
                }

                Array gx2 = 3 * gy1 + gy2;
                Array gx3 = 2 * gy2;
                bctx.input_grad(0) = gx2;  // by index, from non-temporary
                bctx.input_grad(1) = gx3;
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array x2 = x2_value.MakeView().RequireGrad(graph_id);
    Array x3 = x3_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    Array y2{};
    forward(x1, x2, x3, y1, y2);

    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        y1.SetGrad(gy1_value.MakeView().RequireGrad(graph_id), graph_id);
        y2.SetGrad(gy2_value.MakeView().RequireGrad(graph_id), graph_id);
    } else {
        y1.SetGrad(gy1_value, graph_id);
        y2.SetGrad(gy2_value, graph_id);
    }
    Backward({y1, y2}, graph_id, double_backprop_opt);

    testing::ExpectEqual(gy1_value, *y1.GetGrad(graph_id));
    testing::ExpectEqual(gy2_value, *y2.GetGrad(graph_id));
    testing::ExpectEqual(gx1_value, *x1.GetGrad(graph_id));
    testing::ExpectEqual(gx2_value, *x2.GetGrad(graph_id));
    testing::ExpectEqual(gx3_value, *x3.GetGrad(graph_id));
    if (double_backprop_opt == DoubleBackpropOption::kEnable) {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsGradRequired(graph_id));
        EXPECT_TRUE(y2.GetGrad(graph_id)->IsGradRequired(graph_id));
    } else {
        EXPECT_TRUE(y1.GetGrad(graph_id)->IsConstant());
        EXPECT_TRUE(y2.GetGrad(graph_id)->IsConstant());
    }
}

INSTANTIATE_TEST_CASE_P(Params, BackpropFunctionTest, ::testing::Values(DoubleBackpropOption::kDisable, DoubleBackpropOption::kEnable));

TEST(BackpropGradValidationTest, InvalidGradShape) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});

    auto forward = [](const Array& x1, Array& y1) {
        ASSERT_FALSE(x1.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_TRUE(y1.IsConstant());

        {
            BackwardBuilder bb{"func", y1};
            bb.Define({x1}, [](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);
                EXPECT_TRUE(gy1.IsConstant());
                bctx.input_grad() = gy1.Reshape({2, 1});  // Intentionally set to a wrong shape (2, 1), instead of (2,).
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    forward(x1, y1);

    y1.SetGrad(gy1_value, graph_id);

    // The shape of the computed gradient of x1 is (2, 1) but the shape of x1 is (2,), thus an exception should be thrown.
    EXPECT_THROW(Backward({y1}, graph_id, DoubleBackpropOption::kDisable), DimensionError);
}

TEST(BackpropGradValidationTest, InvalidGradDtype) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});

    auto forward = [](const Array& x1, Array& y1) {
        ASSERT_FALSE(x1.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_TRUE(y1.IsConstant());

        {
            BackwardBuilder bb{"func", y1};
            bb.Define({x1}, [](BackwardContext& bctx) {
                const Array& gy1 = bctx.output_grad(0);
                EXPECT_TRUE(gy1.IsConstant());
                bctx.input_grad() = gy1.AsType(Dtype::kFloat32);  // Intentionally set to a wrong dtype float, instead of double.
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    forward(x1, y1);

    y1.SetGrad(gy1_value, graph_id);

    // The dtype of the computed gradient of x1 is float but the dtype of x1 is double, thus an exception should be thrown.
    EXPECT_THROW(Backward({y1}, graph_id, DoubleBackpropOption::kDisable), DtypeError);
}

TEST(BackpropGradValidationTest, InvalidGradDevice) {
    testing::DeviceSession device_session({native::NativeBackend::kDefaultName, 0});

    using T = double;
    GraphId graph_id = "testgraph";
    Shape shape{2};
    Array x1_value = testing::BuildArray(shape).WithData<T>({1, 2});
    Array gy1_value = testing::BuildArray(shape).WithData<T>({1, -3});

    auto forward = [](const Array& x1, Array& y1) {
        ASSERT_FALSE(x1.IsConstant());
        y1 = 2 * x1.AsGradStopped() + 1;
        ASSERT_TRUE(y1.IsConstant());

        {
            BackwardBuilder bb{"func", y1};
            bb.Define({x1}, [& device = x1.device()](BackwardContext & bctx) {
                const Array& gy1 = bctx.output_grad(0);
                EXPECT_TRUE(gy1.IsConstant());
                bctx.input_grad() =
                        gy1.ToDevice(device.backend().GetDevice(device.index() + 1));  // Intentionally set to a different device.
            });
        }
    };

    Array x1 = x1_value.MakeView().RequireGrad(graph_id);
    Array y1{};
    forward(x1, y1);

    y1.SetGrad(gy1_value, graph_id);

    // The device of the computed gradient of x1 is on a different device from the device of x1, thus an exception should be throws.
    EXPECT_THROW(Backward({y1}, graph_id, DoubleBackpropOption::kDisable), DeviceError);
}

}  // namespace
}  // namespace xchainer
