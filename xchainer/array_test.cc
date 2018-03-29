#include "xchainer/array.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array_node.h"
#include "xchainer/backend.h"
#include "xchainer/check_backward.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/op_node.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/context_session.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

public:
    template <typename T>
    void CheckContiguousFill(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(scalar);
        testing::ExpectDataEqual(expected, x);
    }

    template <typename T>
    void CheckContiguousFill(T value) {
        CheckContiguousFill(value, value);
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(ArrayTest, DefaultCtor) {
    Array a;
    EXPECT_EQ(nullptr, a.body().get());
}

TEST_P(ArrayTest, CopyCtor) {
    Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
    Array b = a;  // NOLINT

    // A copy-constructed instance must share the same body.
    EXPECT_EQ(a.body().get(), b.body().get());
}

TEST_P(ArrayTest, ArrayMoveCtor) {
    { EXPECT_TRUE(std::is_nothrow_move_constructible<Array>::value); }

    // A view must not be affected by move
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = a.MakeView();
        Array c = std::move(a);
        testing::ExpectEqual<float>(b, c);
    }

    // A copy must not be affected by move
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = a.Copy();
        Array c = std::move(a);
        testing::ExpectEqualCopy<float>(b, c);
    }

    // Array body must be transferred by move
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        auto body = a.body();
        Array c = std::move(a);
        EXPECT_EQ(body, c.body());
    }
}

TEST_P(ArrayTest, ArrayBodyCtor) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    auto body = a.body();
    Array b{body};
    EXPECT_EQ(body, b.body());
    testing::ExpectArraysEqualAttributes(a, b);
    EXPECT_EQ(a.data(), b.data());
    EXPECT_THROW(internal::GetArrayNode(a), XchainerError);
    EXPECT_THROW(internal::GetArrayNode(b), XchainerError);
}

TEST_P(ArrayTest, CopyAssignment) {
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, true, true});
        Array b;
        b = a;

        EXPECT_EQ(a.body().get(), b.body().get());
    }
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, true, true});
        Array b = testing::BuildArray<float>({1}, {1.0f});
        b = a;

        EXPECT_EQ(a.body().get(), b.body().get());
    }
}

TEST_P(ArrayTest, MoveAssignment) {
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, true, true});
        Array b;
        std::shared_ptr<xchainer::internal::ArrayBody> body = a.body();
        b = std::move(a);

        EXPECT_EQ(body.get(), b.body().get());
    }
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, true, true});
        Array b = testing::BuildArray<float>({1}, {1.0f});
        std::shared_ptr<xchainer::internal::ArrayBody> body = a.body();
        b = std::move(a);

        EXPECT_EQ(body.get(), b.body().get());
    }
}

TEST_P(ArrayTest, SetRequiresGrad) {
    // Default graph
    {
        Array x = testing::BuildArray<bool>({1}, {true});
        ASSERT_FALSE(x.IsGradRequired());
        x.RequireGrad();
        ASSERT_TRUE(x.IsGradRequired());
    }

    // User-specified graph
    {
        GraphId graph_id = "graph_1";
        Array x = testing::BuildArray<bool>({1}, {true});
        ASSERT_FALSE(x.IsGradRequired(graph_id));
        x.RequireGrad(graph_id);
        ASSERT_TRUE(x.IsGradRequired(graph_id));
    }
}

TEST_P(ArrayTest, Grad) {
    GraphId graph_id = "graph_1";
    Shape shape{2, 3};
    using T = float;

    Array x = testing::BuildArray<T>(shape, {5, 3, 2, 1, 4, 6});
    Array g = testing::BuildArray<T>(shape, {8, 4, 6, 3, 2, 1});

    x.RequireGrad(graph_id);
    g.RequireGrad(graph_id);

    EXPECT_FALSE(x.GetGrad(graph_id)) << "grad must be initially unset";

    // Set and get grad
    {
        x.SetGrad(g, graph_id);

        testing::ExpectEqual<T>(g, *x.GetGrad(graph_id));
    }

    // Get grad multiple times
    {
        const nonstd::optional<Array>& grad1 = x.GetGrad(graph_id);
        const nonstd::optional<Array>& grad2 = x.GetGrad(graph_id);
        EXPECT_EQ(&*grad1, &*grad2) << "Multiple retrieval of grad must return the same arrays";
    }

    // ClearGrad
    {
        Array grad_view = *x.GetGrad(graph_id);  // Make a view of grad

        x.ClearGrad(graph_id);

        EXPECT_FALSE(x.GetGrad(graph_id)) << "grad must be cleared after calling ClearGrad()";

        // ClearGrad() must not affect previously retrieved view to grad
        testing::ExpectEqual<T>(grad_view, g);
    }
}

TEST_P(ArrayTest, ArrayFromBuffer) {
    using T = int32_t;
    Shape shape{3, 2};

    std::vector<T> raw_data{0, 1, 2, 3, 4, 5};
    std::shared_ptr<T> data{&raw_data[0], [](const T*) {}};

    Dtype dtype = TypeToDtype<T>;
    Array x = Array::FromBuffer(shape, dtype, data);

    // Basic attributes
    EXPECT_EQ(shape, x.shape());
    EXPECT_EQ(dtype, x.dtype());
    EXPECT_EQ(2, x.ndim());
    EXPECT_EQ(3 * 2, x.GetTotalSize());
    EXPECT_EQ(int64_t{sizeof(T)}, x.element_bytes());
    EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetTotalBytes());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());

    // Array::data
    testing::ExpectDataEqual<T>(data.get(), x);

    Device& device = GetDefaultDevice();
    EXPECT_EQ(&device, &x.device());
    if (device.backend().GetName() == "native") {
        EXPECT_EQ(data.get(), x.data().get());
    } else if (device.backend().GetName() == "cuda") {
        EXPECT_NE(data.get(), x.data().get());
    } else {
        FAIL() << "invalid device_id";
    }
}

TEST_P(ArrayTest, Empty) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Array x = Array::Empty(Shape{3, 2}, dtype);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_EQ(x.shape(), Shape({3, 2}));
    EXPECT_EQ(x.dtype(), dtype);
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, EmptyLike) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Array x_orig = Array::Empty(Shape{3, 2}, dtype);
    Array x = Array::EmptyLike(x_orig);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_NE(x.data(), x_orig.data());
    EXPECT_EQ(x.shape(), x_orig.shape());
    EXPECT_EQ(x.dtype(), x_orig.dtype());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, ContiguousFill) {
    CheckContiguousFill(true);
    CheckContiguousFill(false);
    CheckContiguousFill(int8_t{0});
    CheckContiguousFill(int8_t{-1});
    CheckContiguousFill(int8_t{5});
    CheckContiguousFill(int8_t{-128});
    CheckContiguousFill(int8_t{127});
    CheckContiguousFill(int16_t{0});
    CheckContiguousFill(int16_t{-3});
    CheckContiguousFill(int32_t{0});
    CheckContiguousFill(int32_t{-3});
    CheckContiguousFill(int64_t{0});
    CheckContiguousFill(int64_t{-3});
    CheckContiguousFill(uint8_t{0});
    CheckContiguousFill(uint8_t{255});
    CheckContiguousFill(float{0});
    CheckContiguousFill(float{std::numeric_limits<float>::infinity()});
    CheckContiguousFill(float{std::nanf("")});
    CheckContiguousFill(double{0});
    CheckContiguousFill(double{std::numeric_limits<double>::infinity()});
    CheckContiguousFill(double{std::nan("")});

    CheckContiguousFill(true, Scalar(int32_t{1}));
    CheckContiguousFill(true, Scalar(int32_t{2}));
    CheckContiguousFill(true, Scalar(int32_t{-1}));
    CheckContiguousFill(false, Scalar(int32_t{0}));
    CheckContiguousFill(int8_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int8_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int8_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int8_t{1}, Scalar(true));
    CheckContiguousFill(int8_t{1}, Scalar(1.0f));
    CheckContiguousFill(int8_t{1}, Scalar(1.0));
    CheckContiguousFill(int16_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int16_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int16_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int16_t{1}, Scalar(true));
    CheckContiguousFill(int16_t{1}, Scalar(1.0f));
    CheckContiguousFill(int16_t{1}, Scalar(1.0));
    CheckContiguousFill(int32_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int32_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int32_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int32_t{1}, Scalar(true));
    CheckContiguousFill(int32_t{1}, Scalar(1.0f));
    CheckContiguousFill(int32_t{1}, Scalar(1.0));
    CheckContiguousFill(int64_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int64_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int64_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int64_t{1}, Scalar(true));
    CheckContiguousFill(int64_t{1}, Scalar(1.0f));
    CheckContiguousFill(int64_t{1}, Scalar(1.0));
    CheckContiguousFill(uint8_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(uint8_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(uint8_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(uint8_t{1}, Scalar(true));
    CheckContiguousFill(uint8_t{1}, Scalar(1.0f));
    CheckContiguousFill(uint8_t{1}, Scalar(1.0));
    CheckContiguousFill(float{1}, Scalar(int32_t{1}));
    CheckContiguousFill(float{1}, Scalar(int64_t{1}));
    CheckContiguousFill(float{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(float{1}, Scalar(true));
    CheckContiguousFill(float{1}, Scalar(1.0f));
    CheckContiguousFill(float{1}, Scalar(1.0));
    CheckContiguousFill(double{1}, Scalar(int32_t{1}));
    CheckContiguousFill(double{1}, Scalar(int64_t{1}));
    CheckContiguousFill(double{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(double{1}, Scalar(true));
    CheckContiguousFill(double{1}, Scalar(1.0f));
    CheckContiguousFill(double{1}, Scalar(1.0));
}

TEST_P(ArrayTest, NonContiguousFill) {
    Dtype dtype = Dtype::kFloat32;
    float value = 1.0f;
    {
        Array a = Array::Zeros(Shape{3, 3}, dtype);
        Array b = a.Transpose();
        b.Fill(value);
        testing::ExpectDataEqual(value, b);
        testing::ExpectDataEqual(value, a);
    }
    {
        Array a = Array::Zeros(Shape{3, 3}, dtype);
        a.At({1}).Fill(value);
        testing::ExpectDataEqual(value, a.At({1}));
        // check other rows are not affected
        testing::ExpectDataEqual(0.0f, a.At({0}));
        testing::ExpectDataEqual(0.0f, a.At({2}));
    }
    {
        Array a = Array::Zeros(Shape{3, 3}, dtype);
        a.At({Slice{}, {1}}).Fill(value);
        testing::ExpectDataEqual(value, a.At({Slice{}, {1}}));
        // check other columns are not affected
        testing::ExpectDataEqual(0.0f, a.At({Slice{}, {0}}));
        testing::ExpectDataEqual(0.0f, a.At({Slice{}, {2}}));
    }
}

TEST_P(ArrayTest, FullWithGivenDtype) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Scalar scalar{int64_t{3}};
    Array x = Array::Full(Shape{3, 2}, scalar, dtype);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_EQ(x.shape(), Shape({3, 2}));
    EXPECT_EQ(x.dtype(), dtype);
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{3}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, FullWithScalarDtype) {
    using T = int32_t;
    Scalar scalar{T{3}};
    Array x = Array::Full(Shape{3, 2}, scalar);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_EQ(x.shape(), Shape({3, 2}));
    EXPECT_EQ(x.dtype(), scalar.dtype());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{3}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, FullLike) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Scalar scalar{int64_t{3}};
    Array x_orig = Array::Empty(Shape{3, 2}, dtype);
    Array x = Array::FullLike(x_orig, scalar);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_NE(x.data(), x_orig.data());
    EXPECT_EQ(x.shape(), x_orig.shape());
    EXPECT_EQ(x.dtype(), x_orig.dtype());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{3}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, Zeros) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Array x = Array::Zeros(Shape{3, 2}, dtype);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_EQ(x.shape(), Shape({3, 2}));
    EXPECT_EQ(x.dtype(), dtype);
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{0}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, ZerosLike) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Array x_orig = Array::Empty(Shape{3, 2}, dtype);
    Array x = Array::ZerosLike(x_orig);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_NE(x.data(), x_orig.data());
    EXPECT_EQ(x.shape(), x_orig.shape());
    EXPECT_EQ(x.dtype(), x_orig.dtype());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{0}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, Ones) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Array x = Array::Ones(Shape{3, 2}, dtype);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_EQ(x.shape(), Shape({3, 2}));
    EXPECT_EQ(x.dtype(), dtype);
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{1}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, OnesLike) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Array x_orig = Array::Empty(Shape{3, 2}, dtype);
    Array x = Array::OnesLike(x_orig);
    EXPECT_NE(x.data(), nullptr);
    EXPECT_NE(x.data(), x_orig.data());
    EXPECT_EQ(x.shape(), x_orig.shape());
    EXPECT_EQ(x.dtype(), x_orig.dtype());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());
    testing::ExpectDataEqual(T{1}, x);
    EXPECT_EQ(&GetDefaultDevice(), &x.device());
}

TEST_P(ArrayTest, Equality) {
    using T = float;
    Shape shape{2};
    Array a = testing::BuildArray(shape).WithData<T>({1.0f, 2.0f});
    Array b = testing::BuildArray(shape).WithData<T>({-1.0f, 2.0f});
    Array e = testing::BuildArray(shape).WithData<bool>({false, true});
    Array c = a == b;

    ASSERT_EQ(c.dtype(), Dtype::kBool);
    testing::ExpectEqual<bool>(e, c);
}

TEST_P(ArrayTest, IAdd) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
    a += b;
    testing::ExpectEqual<float>(e, a);
}

TEST_P(ArrayTest, IMul) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {1, 4, 9});
    a *= b;
    testing::ExpectEqual<float>(e, a);
}

TEST_P(ArrayTest, Add) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
    Array o = a + b;
    testing::ExpectEqual<float>(e, o);
}

TEST_P(ArrayTest, Mul) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {1, 4, 9});
    Array o = a * b;
    testing::ExpectEqual<float>(e, o);
}

// TODO(hvy): Also test CUDA using ArrayTest.
TEST(ArrayNativeTest, MulScalar) {
    testing::ContextSession context_session;

    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});

    {
        Array o = a * Scalar{2.f};
        testing::ExpectEqual<float>(e, o);
    }
    {
        Array o = Scalar(2.f) * a;
        testing::ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ComputationalGraph) {
    // c = a + b
    // o = a * c
    Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});

    GraphId graph_id = "graph_1";
    a.RequireGrad(graph_id);
    b.RequireGrad(graph_id);

    {
        auto a_node = internal::GetArrayNode(a, graph_id);
        auto b_node = internal::GetArrayNode(b, graph_id);
        EXPECT_NE(a_node, nullptr);
        EXPECT_NE(b_node, nullptr);
        auto a_op_node = a_node->next_node();
        auto b_op_node = b_node->next_node();
        EXPECT_EQ(a_op_node, nullptr);
        EXPECT_EQ(b_op_node, nullptr);
    }

    Array c = a + b;
    {
        auto a_node = internal::GetArrayNode(a, graph_id);
        auto b_node = internal::GetArrayNode(b, graph_id);
        auto c_node = internal::GetArrayNode(c, graph_id);
        EXPECT_NE(a_node, nullptr);
        EXPECT_NE(b_node, nullptr);
        EXPECT_NE(c_node, nullptr);
        auto a_op_node = a_node->next_node();
        auto b_op_node = b_node->next_node();
        auto c_op_node = c_node->next_node();
        EXPECT_EQ(a_op_node, nullptr);
        EXPECT_EQ(b_op_node, nullptr);
        EXPECT_NE(c_op_node, nullptr);
        EXPECT_EQ(c_op_node->name(), "add");
    }

    Array o = a * c;
    {
        auto a_node = internal::GetArrayNode(a, graph_id);
        auto b_node = internal::GetArrayNode(b, graph_id);
        auto c_node = internal::GetArrayNode(c, graph_id);
        auto o_node = internal::GetArrayNode(o, graph_id);
        EXPECT_NE(a_node, nullptr);
        EXPECT_NE(b_node, nullptr);
        EXPECT_NE(c_node, nullptr);
        EXPECT_NE(o_node, nullptr);
        auto a_op_node = a_node->next_node();
        auto b_op_node = b_node->next_node();
        auto c_op_node = c_node->next_node();
        auto o_op_node = o_node->next_node();
        EXPECT_EQ(a_op_node, nullptr);
        EXPECT_EQ(b_op_node, nullptr);
        EXPECT_NE(c_op_node, nullptr);
        EXPECT_NE(o_op_node, nullptr);
        EXPECT_EQ(c_op_node->name(), "add");
        EXPECT_EQ(o_op_node->name(), "mul");
    }
}

TEST_P(ArrayTest, InplaceNotAllowedWithRequiresGrad) {
    GraphId graph_id = "graph_1";
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});
        a.RequireGrad(graph_id);
        b.RequireGrad(graph_id);
        EXPECT_THROW({ a += b; }, XchainerError);
    }

    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});
        a.RequireGrad(graph_id);
        b.RequireGrad(graph_id);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }

    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});
        a.RequireGrad(graph_id);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }

    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});
        b.RequireGrad(graph_id);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }
}

TEST_P(ArrayTest, Transpose) {
    Array a = testing::BuildArray({2, 3})         //
                      .WithLinearData<int32_t>()  //
                      .WithPadding(0);
    Array b = a.Transpose();

    EXPECT_EQ(Shape({3, 2}), b.shape());
    EXPECT_EQ(Strides({4, 12}), b.strides());

    Array e = testing::BuildArray({3, 2}).WithData<int32_t>({0, 3, 1, 4, 2, 5});
    testing::ExpectEqual<int32_t>(e, b);
}

TEST_P(ArrayTest, Copy) {
    using T = int32_t;
    Array a = testing::BuildArray<T>({3, 1}, {1, 2, 3});
    Array o = a.Copy();
    testing::ExpectEqualCopy<T>(a, o);
}

TEST_P(ArrayTest, MakeView) {
    Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
    Array o = a.MakeView();
    testing::ExpectEqualView<bool>(a, o);
}

TEST_P(ArrayTest, AsConstantCopy) {
    // Stop gradients on all graphs
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        Array b = a.AsConstant(CopyKind::kCopy);

        EXPECT_EQ(&b.device(), &a.device());

        testing::ExpectEqualCopy<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
    }

    // Stop gradients on graphs
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        a.RequireGrad("graph_3");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        ASSERT_TRUE(a.IsGradRequired("graph_3"));
        Array b = a.AsConstant({"graph_1", "graph_2"}, CopyKind::kCopy);

        EXPECT_EQ(&b.device(), &a.device());

        testing::ExpectEqualCopy<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));
        EXPECT_TRUE(b.IsGradRequired("graph_3"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
        EXPECT_TRUE(a.IsGradRequired("graph_3"));
    }

    // Non-contiguous
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false})  //
                          .WithPadding(4);
        Array b = a.AsConstant(CopyKind::kCopy);
        EXPECT_EQ(&b.device(), &a.device());
        testing::ExpectEqualCopy<bool>(a, b);
    }
}

TEST_P(ArrayTest, AsConstantView) {
    // Stop gradients on all graphs
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        Array b = a.AsConstant();

        testing::ExpectEqualView<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
    }

    // Stop gradients on some graphs
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        a.RequireGrad("graph_3");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        ASSERT_TRUE(a.IsGradRequired("graph_3"));
        Array b = a.AsConstant({"graph_1", "graph_2"});

        testing::ExpectEqualView<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));
        EXPECT_TRUE(b.IsGradRequired("graph_3"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
        EXPECT_TRUE(a.IsGradRequired("graph_3"));
    }
    // Non-contiguous
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false})  //
                          .WithPadding(4);
        Array b = a.AsConstant(CopyKind::kView);
        EXPECT_EQ(&b.device(), &a.device());
        testing::ExpectEqualView<bool>(a, b);
    }
}

TEST_P(ArrayTest, AddBackward) {
    Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});

    a.RequireGrad();
    b.RequireGrad();

    Array o = a + b;

    auto op_node = internal::GetArrayNode(o)->next_node();
    Array go = testing::BuildArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go, {kDefaultGraphId});
    Array gb = op_node->backward_functions()[1](go, {kDefaultGraphId});

    testing::ExpectEqual<bool>(ga, go);
    testing::ExpectEqual<bool>(gb, go);
}

TEST_P(ArrayTest, MulBackward) {
    Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});

    a.RequireGrad();
    b.RequireGrad();

    Array o = a * b;

    auto op_node = internal::GetArrayNode(o)->next_node();
    Array go = testing::BuildArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go, {kDefaultGraphId});
    Array gb = op_node->backward_functions()[1](go, {kDefaultGraphId});

    testing::ExpectEqual<bool>(ga, go * b);
    testing::ExpectEqual<bool>(gb, go * a);

    EXPECT_FALSE(ga.IsGradRequired());
    EXPECT_FALSE(gb.IsGradRequired());
}

TEST_P(ArrayTest, MulBackwardCapture) {
    Array y = [this]() {
        Array x1 = testing::BuildArray<float>({1}, {2.0f});
        Array x2 = testing::BuildArray<float>({1}, {3.0f});
        x1.RequireGrad();
        x2.RequireGrad();
        return x1 * x2;
    }();
    auto op_node = internal::GetArrayNode(y)->next_node();
    auto lhs_func = op_node->backward_functions()[0];
    auto rhs_func = op_node->backward_functions()[1];
    Array gy = testing::BuildArray<float>({1}, {1.0f});

    Array gx1 = lhs_func(gy, {kDefaultGraphId});
    Array e1 = testing::BuildArray<float>({1}, {3.0f});
    testing::ExpectEqual<float>(e1, gx1);
    EXPECT_FALSE(gx1.IsGradRequired());

    Array gx2 = rhs_func(gy, {kDefaultGraphId});
    Array e2 = testing::BuildArray<float>({1}, {2.0f});
    testing::ExpectEqual<float>(e2, gx2);
    EXPECT_FALSE(gx2.IsGradRequired());
}

TEST_P(ArrayTest, MulBackwardMultipleGraphs) {
    GraphId graph_id1 = "graph_1";
    GraphId graph_id2 = "graph_2";

    Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::BuildArray<bool>({4, 1}, {true, false, true, false});

    a.RequireGrad(graph_id1);
    b.RequireGrad(graph_id2);

    Array o = a * b;
    Array go = testing::BuildArray<bool>({4, 1}, {true, true, true, true});

    auto op_node1 = internal::GetArrayNode(o, graph_id1)->next_node();
    Array ga = op_node1->backward_functions()[0](go, {graph_id1});

    auto op_node2 = internal::GetArrayNode(o, graph_id2)->next_node();
    Array gb = op_node2->backward_functions()[0](go, {graph_id2});

    EXPECT_FALSE(ga.IsGradRequired(graph_id1));
    EXPECT_TRUE(ga.IsGradRequired(graph_id2));

    EXPECT_TRUE(gb.IsGradRequired(graph_id1));
    EXPECT_FALSE(gb.IsGradRequired(graph_id2));
}

TEST_P(ArrayTest, MultipleGraphsRequireGradDefault) {
    Array a = testing::BuildArray<float>({1}, {2.0f});

    EXPECT_FALSE(a.IsGradRequired());

    a.RequireGrad();

    EXPECT_TRUE(a.IsGradRequired());
    EXPECT_THROW(a.RequireGrad(), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsRequireGradNamed) {
    GraphId graph_id = "graph_1";

    Array a = testing::BuildArray<float>({1}, {2.0f});

    ASSERT_FALSE(a.IsGradRequired(graph_id));

    a.RequireGrad(graph_id);

    EXPECT_TRUE(a.IsGradRequired(graph_id));
    EXPECT_THROW(a.RequireGrad(graph_id), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsRequireGradChainedCallsCtor) {
    Array a = (*testing::BuildArray<float>({1}, {2.0f})).RequireGrad();

    EXPECT_TRUE(a.IsGradRequired());
    EXPECT_THROW(a.RequireGrad(), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsRequireGradChainedCallsRequireGrad) {
    Array a = testing::BuildArray<float>({1}, {2.0f});

    EXPECT_THROW(a.RequireGrad().RequireGrad(), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsForward) {
    Array a = testing::BuildArray<float>({1}, {2.0f});
    Array b = testing::BuildArray<float>({1}, {2.0f});

    GraphId graph_id_1 = "graph_1";
    GraphId graph_id_2 = "graph_2";

    a.RequireGrad(graph_id_1);
    b.RequireGrad(graph_id_2);

    EXPECT_TRUE(a.IsGradRequired(graph_id_1));
    EXPECT_FALSE(a.IsGradRequired(graph_id_2));

    EXPECT_FALSE(b.IsGradRequired(graph_id_1));
    EXPECT_TRUE(b.IsGradRequired(graph_id_2));

    Array o = a * b;

    EXPECT_TRUE(o.IsGradRequired(graph_id_1));
    EXPECT_TRUE(o.IsGradRequired(graph_id_2));

    // No unspecified graphs are generated
    EXPECT_FALSE(o.IsGradRequired(kDefaultGraphId));
    EXPECT_FALSE(o.IsGradRequired("graph_3"));
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ArrayTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

TEST(ArrayAtTest, At) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 1};
    Shape output_shape{1, 2, 1};
    std::vector<ArrayIndex> indices{-1, NewAxis{}, Slice{1, 3}};
    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array b = a.At(indices);

    EXPECT_EQ(output_shape, b.shape());
    Array e = testing::BuildArray(output_shape).WithData<T>({4, 5});
    testing::ExpectEqual<T>(e, b);

    // Check if strides are 0 for newaxis.
    EXPECT_EQ(0, b.strides()[0]);
    EXPECT_NE(0, b.strides()[1]);
    EXPECT_NE(0, b.strides()[2]);
}

TEST(ArrayReshapeTest, Reshape) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, 4, 2};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array b = a.Reshape(output_shape);
    ASSERT_EQ(output_shape, b.shape());
    EXPECT_EQ(a.data().get(), b.data().get()) << "Reshape must be done without copying data";
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();
    testing::ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeSpecifiedUnitLenghtAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array b = a.Squeeze(std::vector<int8_t>{2, 0, 4});
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();
    testing::ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeAllAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::BuildArray({1, 1, 1}).WithLinearData<T>();
    Array b = a.Squeeze();
    Array e = testing::BuildArray<T>({}, std::vector<T>(1, 0));
    testing::ExpectEqual<T>(e, b);
}

TEST(ArrayBroadcastToTest, BroadcastTo) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 1};
    Shape output_shape{3, 1, 2, 3, 1, 2};

    Array aa = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array a = aa.At({Slice(), Slice(), Slice(), NewAxis{}});  // Make a broadcastable axis.
    ASSERT_EQ(Shape({2, 3, 1, 1}), a.shape());                // Check test precondition

    Array b = a.BroadcastTo(output_shape);
    ASSERT_EQ(output_shape, b.shape());
    EXPECT_EQ(a.data().get(), b.data().get()) << "BroadcastTo must be done without copying data";
    ASSERT_EQ(0, b.strides()[1]) << "Stride of broadcasted dimension must be 0";

    std::vector<int64_t> output_data;
    for (int i = 0; i < 3; ++i) {
        output_data.insert(output_data.end(), {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    }
    Array e = testing::BuildArray(output_shape).WithData<T>(output_data.begin(), output_data.end());
    testing::ExpectEqual<T>(e, b);
}

TEST(ArraySumTest, Sum) {
    using T = float;
    testing::ContextSession context_session{};

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum(std::vector<int8_t>{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::BuildArray(Shape{2}).WithData<T>({630.0f, 1926.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST(ArraySumTest, SumAllAxes) {
    using T = float;
    testing::ContextSession context_session{};

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum();
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({153.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST(ArraySumTest, SumKeepDims) {
    using T = float;
    testing::ContextSession context_session{};

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum(std::vector<int8_t>{-1, 1}, true);
    EXPECT_EQ(Shape({2, 1, 2, 1}), b.shape());
    EXPECT_EQ(0, b.strides()[1]);
    EXPECT_EQ(0, b.strides()[3]);
    Array e = testing::BuildArray(Shape{2, 1, 2, 1}).WithData<T>({114.0f, 162.0f, 402.0f, 450.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST(ArrayDotTest, Dot) {
    using T = float;
    testing::ContextSession context_session{};

    Array a = testing::BuildArray({2, 3}).WithLinearData(1.f).WithPadding(1);
    Array b = testing::BuildArray<float>({3, 2}, {1.f, 2.f, -1.f, -3.f, 2.f, 4.f}).WithPadding(2);
    Array c = a.Dot(b);
    Array e = testing::BuildArray<float>({2, 2}, {5.f, 8.f, 11.f, 17.f});
    testing::ExpectEqual<float>(e, c);
}

}  // namespace
}  // namespace xchainer
