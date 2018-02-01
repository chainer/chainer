#include "xchainer/backprop.h"
#include <algorithm>

#include <string>
#include <vector>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/op_node.h"
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
    Array MakeFullArray(const Shape& shape, float value) const { return Array::Full(shape, value); }

    std::vector<Array> MakeFullArrays(const Shape& shape, const std::vector<float>& values) const {
        std::vector<Array> ret;
        for (float value : values) {
            ret.push_back(Array::Full(shape, value));
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
    template <typename Fprop, typename... Args>
    void CheckBackpropImpl(std::vector<Array>& target_inputs, std::vector<Array>& expected_grads, Fprop&& fprop, Args&&... args) const {
        ASSERT_EQ(expected_grads.size(), target_inputs.size());

        std::for_each(target_inputs.begin(), target_inputs.end(), [](auto& x) { x.RequireGrad(); });

        auto y = fprop(target_inputs, args...);
        Backward(y);
        for (size_t i = 0; i < expected_grads.size(); ++i) {
            ExpectEqual<float>(expected_grads[i], *target_inputs[i].GetGrad());
        }
        EXPECT_TRUE(y.GetGrad().has_value());
    }

    template <typename Fprop>
    void CheckBackprop(std::vector<Array>& target_inputs, std::vector<Array>& expected_grads, Fprop&& fprop) const {
        CheckBackpropImpl(target_inputs, expected_grads, fprop);
    }

    template <typename Fprop>
    void CheckBackpropExtraInputs(std::vector<Array>& target_inputs, std::vector<Array>& other_inputs, std::vector<Array>& expected_grads,
                                  Fprop&& fprop) const {
        CheckBackpropImpl(target_inputs, expected_grads, fprop, other_inputs);
        for (size_t i = 0; i < other_inputs.size(); ++i) {
            EXPECT_THROW(other_inputs[i].GetGrad(), XchainerError);
        }
    }

    // Simple versions. It makes and uses an array with one element for each input.
    template <typename Fprop>
    void CheckBackpropSingleElement(std::vector<float> target_inputs, std::vector<float> expected_grads, Fprop&& fprop) const {
        auto xs = MakeFullArrays({1}, target_inputs);
        auto expected_gxs = MakeFullArrays({1}, expected_grads);
        CheckBackprop(xs, expected_gxs, std::forward<Fprop>(fprop));
    }

    template <typename Fprop>
    void CheckBackpropSingleElementExtraInputs(std::vector<float> target_inputs, std::vector<float> other_inputs,
                                               std::vector<float> expected_grads, Fprop&& fprop) const {
        auto xs = MakeFullArrays({1}, target_inputs);
        auto other_xs = MakeFullArrays({1}, other_inputs);
        auto expected_gxs = MakeFullArrays({1}, expected_grads);
        CheckBackpropExtraInputs(xs, other_xs, expected_gxs, std::forward<Fprop>(fprop));
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
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

TEST_P(BackpropTest, BackwardSoleArrayNode) {
    auto x = Array::Full({1}, 2.0f);
    x.RequireGrad();
    Backward(x);
    auto e = Array::OnesLike(x);
    ExpectEqual<float>(e, *x.GetGrad());
}

TEST_P(BackpropTest, DoubleBackprop) {
    auto fprop = [](auto& xs, auto& ys) {
        auto z = xs[0] * (xs[0] + ys[0]);
        Backward(z, kDefaultGraphId, DoubleBackpropOption::kEnable);
        auto gx = *xs[0].GetGrad();  // 2x + y
        xs[0].ClearGrad();
        return gx * xs[0];
    };
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {7.0f}, fprop);
}

TEST_P(BackpropTest, MultipleGraphsDoubleBackprop) {
    GraphId graph_x = "graph_x";
    GraphId graph_y = "graph_y";

    auto x = Array::Full({1}, 2.0f);
    x.RequireGrad(graph_x);

    auto y = Array::Full({1}, 3.0f);
    y.RequireGrad(graph_y);

    auto z = x * (x + y);
    Backward(z, graph_x);

    auto gx = *x.GetGrad(graph_x);  // 2x + y
    EXPECT_FALSE(gx.IsGradRequired(graph_x));
    EXPECT_TRUE(gx.IsGradRequired(graph_y));

    auto w = x * gx;
    Backward(w, graph_y);

    auto e = Array::Full({1}, 2.0f);
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
        xs[0].SetGrad(Array::OnesLike(xs[0]));
        return xs[0].Copy();
    };
    CheckBackpropSingleElement({1.0f}, {2.0f}, fprop);
}

TEST_P(BackpropTest, BackwardGivenOutputGrad) {
    auto fprop = [](auto& xs, auto& ys) {
        auto z = xs[0] * ys[0];
        z.SetGrad(Array::FullLike(z, 2.0f));
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

INSTANTIATE_TEST_CASE_P(ForEachDevice, BackpropTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                         std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                         std::string{"cpu"}));

TEST(BackpropEnableDoubleBackpropTest, Enabled) {
    Array x1 = Array::Full({2}, 1.f).RequireGrad();
    Array x2 = Array::Full({2}, 2.f);
    Array y1 = x1 + x2;
    Array y2 = x1 * x2;
    Array z = y1 * y2;
    Backward(z, kDefaultGraphId, DoubleBackpropOption::kEnable);

    std::shared_ptr<const ArrayNode> z_node = internal::GetArrayNode(z);
    ASSERT_TRUE(z_node);

    std::shared_ptr<const OpNode> z_op = z_node->next_node();
    ASSERT_TRUE(z_op);

    auto y_nodes = z_op->next_nodes();
    ASSERT_EQ(2u, y_nodes.size());
    EXPECT_EQ(2u, z_op->backward_functions().size());

    for (const std::shared_ptr<ArrayNode>& y_node : y_nodes) {
        std::shared_ptr<const OpNode> y_op = y_node->next_node();
        ASSERT_TRUE(y_op);
        ASSERT_EQ(1u, y_op->next_nodes().size());
        EXPECT_EQ(1u, y_op->backward_functions().size());
    }
}

TEST(BackpropEnableDoubleBackpropTest, Disabled) {
    Array x1 = Array::Full({2}, 1.f).RequireGrad();
    Array x2 = Array::Full({2}, 2.f);
    Array y1 = x1 + x2;
    Array y2 = x1 * x2;
    Array z = y1 * y2;
    std::shared_ptr<const ArrayNode> z_node = internal::GetArrayNode(z);
    ASSERT_TRUE(z_node);
    std::shared_ptr<const OpNode> z_op = z_node->next_node();
    ASSERT_TRUE(z_op);

    Backward(z);

    ASSERT_TRUE(z_node);
    EXPECT_FALSE(z_node->next_node());

    EXPECT_EQ(0u, z_op->next_nodes().size());
    EXPECT_EQ(0u, z_op->backward_functions().size());
}

}  // namespace
}  // namespace xchainer
