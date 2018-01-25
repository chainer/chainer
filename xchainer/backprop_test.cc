#include "xchainer/backprop.h"

#include <string>
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

class BackpropTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    Array MakeFullArray(const Shape& shape, float value) const { return Array::Full(shape, value); }

    Array MakeFullArray(const Shape& shape, float value, const GraphId& graph_id) const {
        return Array::Full(shape, value).RequireGrad(graph_id);
    }

    std::vector<Array> MakeFullArrays(const Shape& shape, const std::vector<float>& values) const {
        std::vector<Array> ret;
        for (float value : values) {
            ret.push_back(Array::Full(shape, value));
        }
        return ret;
    }

    std::vector<Array> MakeFullArrays(const Shape& shape, const std::vector<float>& values, const GraphId& graph_id) const {
        std::vector<Array> ret;
        for (float value : values) {
            ret.push_back(Array::Full(shape, value));
            ret.back().RequireGrad(graph_id);
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
    void CheckBackpropImpl(std::vector<Array>& target_inputs, std::vector<Array>& expected_grads, Fprop&& fprop, const GraphId& graph_id,
                           Args&&... args) const {
        ASSERT_EQ(expected_grads.size(), target_inputs.size());
        auto y = fprop(target_inputs, args...);
        Backward(y, graph_id);
        for (size_t i = 0; i < expected_grads.size(); ++i) {
            ExpectEqual<float>(expected_grads[i], target_inputs[i].Grad(graph_id));
        }
    }

    template <typename Fprop>
    void CheckBackprop(std::vector<Array>& target_inputs, std::vector<Array>& expected_grads, Fprop&& fprop,
                       const GraphId& graph_id) const {
        CheckBackpropImpl(target_inputs, expected_grads, fprop, graph_id);
    }

    template <typename Fprop>
    void CheckBackpropExtraInputs(std::vector<Array>& target_inputs, std::vector<Array>& other_inputs, std::vector<Array>& expected_grads,
                                  Fprop&& fprop, const GraphId& graph_id) const {
        CheckBackpropImpl(target_inputs, expected_grads, fprop, graph_id, other_inputs);
        for (size_t i = 0; i < other_inputs.size(); ++i) {
            EXPECT_THROW(other_inputs[i].Grad(graph_id), XchainerError);
        }
    }

    // Simple versions. It makes and uses an array with one element for each input.
    template <typename Fprop>
    void CheckBackpropSingleElement(std::vector<float> target_inputs, std::vector<float> expected_grads, Fprop&& fprop,
                                    const GraphId& graph_id) const {
        auto xs = MakeFullArrays({1}, target_inputs, graph_id);
        auto expected_gxs = MakeFullArrays({1}, expected_grads);
        CheckBackprop(xs, expected_gxs, std::forward<Fprop>(fprop), graph_id);
    }

    template <typename Fprop>
    void CheckBackpropSingleElementExtraInputs(std::vector<float> target_inputs, std::vector<float> other_inputs,
                                               std::vector<float> expected_grads, Fprop&& fprop, const GraphId& graph_id) const {
        auto xs = MakeFullArrays({1}, target_inputs, graph_id);
        auto other_xs = MakeFullArrays({1}, other_inputs);
        auto expected_gxs = MakeFullArrays({1}, expected_grads);
        CheckBackpropExtraInputs(xs, other_xs, expected_gxs, std::forward<Fprop>(fprop), graph_id);
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(BackpropTest, BackwardBasic) {
    CheckBackpropSingleElement({3.0f, 2.0f}, {2.0f, 3.0f}, [](auto& xs) { return xs[0] * xs[1]; }, "graph_1");
    CheckBackpropSingleElement({3.0f, 2.0f, 4.0f}, {8.0f, 12.0f, 6.0f}, [](auto& xs) { return (xs[0] * xs[1]) * xs[2]; }, "graph_1");
    CheckBackpropSingleElement({3.0f, 2.0f}, {12.0f, 9.0f}, [](auto& xs) { return (xs[0] * xs[1]) * xs[0]; }, "graph_1");
    CheckBackpropSingleElement({3.0f, 2.0f}, {1.0f, 2.0f}, [](auto& xs) { return (xs[0] + xs[1]) + xs[1]; }, "graph_1");
}

TEST_P(BackpropTest, BackwardWithExtraInputs) {
    CheckBackpropSingleElementExtraInputs({2.0f, 3.0f}, {4.0f}, {3.0f, 6.0f}, [](auto& xs, auto& ys) { return xs[1] * (xs[0] + ys[0]); },
                                          "graph_1");
    CheckBackpropSingleElementExtraInputs({2.0f}, {4.0f}, {4.0f}, [](auto& xs, auto& ys) { return xs[0] * ys[0]; }, "graph_1");
}

TEST_P(BackpropTest, BackwardSoleArrayNode) {
    const GraphId graph_id = "graph_1";
    auto x = Array::Full({1}, 2.0f);
    x.RequireGrad(graph_id);
    Backward(x, graph_id);
    auto e = Array::OnesLike(x);
    ExpectEqual<float>(e, x.Grad(graph_id));
}

TEST_P(BackpropTest, DoubleBackprop) {
    const GraphId graph_id = "graph_1";
    auto fprop = [&graph_id](auto& xs, auto& ys) {
        auto z = xs[0] * (xs[0] + ys[0]);
        Backward(z, graph_id);
        auto gx = xs[0].Grad(graph_id);
        xs[0].ClearGrad(graph_id);
        return gx;
    };
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {2.0f}, fprop, graph_id);
}

TEST_P(BackpropTest, BackwardInputToMultipleOps) {
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {7.0f}, [](auto& xs, auto& ys) { return xs[0] * (xs[0] + ys[0]); }, "graph_1");
}

TEST_P(BackpropTest, BackwardIdenticalInputs) {
    CheckBackpropSingleElement({2.0f}, {2.0f}, [](auto& xs) { return xs[0] + xs[0]; }, "graph_1");
    CheckBackpropSingleElement({3.0f}, {6.0f}, [](auto& xs) { return xs[0] * xs[0]; }, "graph_1");
}

TEST_P(BackpropTest, BackwardGivenInputGrad) {
    const GraphId& graph_id = "graph_1";
    auto fprop = [graph_id](auto& xs) {
        xs[0].SetGrad(Array::OnesLike(xs[0]), graph_id);
        return xs[0];
    };
    CheckBackpropSingleElement({1.0f}, {2.0f}, fprop, graph_id);
}

TEST_P(BackpropTest, BackwardGivenOutputGrad) {
    const GraphId& graph_id = "graph_1";
    auto fprop = [graph_id](auto& xs, auto& ys) {
        auto z = xs[0] * ys[0];
        z.SetGrad(Array::FullLike(z, 2.0f), graph_id);
        return z;
    };
    CheckBackpropSingleElementExtraInputs({2.0f}, {3.0f}, {6.0f}, fprop, graph_id);
}

// TODO(hvy): Clean up tests
TEST_P(BackpropTest, MultipleGraphsReuse) {
    GraphId graph_id_1 = "graph_1";
    GraphId graph_id_2 = "graph_2";

    Array x1 = MakeFullArray({1}, {2.0f});
    Array x2 = MakeFullArray({1}, {5.0f});

    EXPECT_FALSE(x1.RequiresGrad(graph_id_1));
    EXPECT_FALSE(x2.RequiresGrad(graph_id_2));

    x1.RequireGrad(graph_id_1);
    x2.RequireGrad(graph_id_2);

    EXPECT_TRUE(x1.RequiresGrad(graph_id_1));
    EXPECT_TRUE(x2.RequiresGrad(graph_id_2));

    Array y1 = x1 * x2;
    Backward(y1, graph_id_1);

    Array expected_1 = MakeFullArray({1}, {5.0f});
    ExpectEqual<float>(expected_1, x1.Grad(graph_id_1));
    ASSERT_THROW(x2.Grad(graph_id_2), XchainerError);

    x1.ClearGrad(graph_id_1);
    x2.ClearGrad(graph_id_2);

    Array y2 = x1 * x2;
    Backward(y2, graph_id_2);

    Array expected_2 = MakeFullArray({1}, {2.0f});
    ExpectEqual<float>(expected_2, x2.Grad(graph_id_2));
    ASSERT_THROW(x1.Grad(graph_id_1), XchainerError);

    x1.ClearGrad(graph_id_1);
    x2.ClearGrad(graph_id_2);

    x1.RequireGrad(graph_id_2);

    Array y3 = x1 * x2;
    Backward(y3, graph_id_2);

    ExpectEqual<float>(expected_1, x1.Grad(graph_id_2));
    ExpectEqual<float>(expected_2, x2.Grad(graph_id_2));
    ASSERT_THROW(x1.Grad(graph_id_1), XchainerError);
    ASSERT_THROW(x2.Grad(graph_id_1), XchainerError);
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, BackpropTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                         std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                         std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
