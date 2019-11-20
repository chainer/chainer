#include "chainerx/backward_builder.h"

#include <utility>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/backward.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/context_session.h"

// TODO(niboshi): Move test cases related to BackwardBuilder to this file (from backward_test.cc).

namespace chainerx {
namespace {

TEST(BackwardBuilderTest, FloatToInt_NotBackproppable) {
    testing::ContextSession context_session;

    auto forward = [](const Array& x, Array& y1, Array& y2) {
        y1 = Ones({2, 3}, Dtype::kInt32, x.device());
        y2 = Ones({2, 3}, Dtype::kInt32, x.device());

        BackwardBuilder bb{"forward", x, {y1, y2}};
        BackwardBuilder::Target bt = bb.CreateTarget();
        EXPECT_FALSE(static_cast<bool>(bt));
        bb.Finalize();
    };

    Array x = Ones({2, 3}, Dtype::kFloat32).RequireGrad();
    Array y1{};
    Array y2{};
    forward(x, y1, y2);
    EXPECT_FALSE(y1.IsBackpropRequired());
    EXPECT_FALSE(y2.IsBackpropRequired());
}

TEST(BackwardBuilderTest, FloatToInt_PartiallyBackproppable) {
    testing::ContextSession context_session;

    auto forward = [](const Array& x, Array& y1, Array& y2) {
        y1 = Ones({2, 3}, Dtype::kInt32, x.device());
        y2 = Ones({2, 3}, Dtype::kFloat32, x.device());

        BackwardBuilder bb{"forward", x, {y1, y2}};
        BackwardBuilder::Target bt = bb.CreateTarget();
        EXPECT_TRUE(static_cast<bool>(bt));
        bt.Define([](BackwardContext& bctx) {
            EXPECT_EQ(bctx.input_count(), 1U);
            EXPECT_EQ(bctx.output_count(), 2U);
            bctx.input_grad() = *bctx.output_grad(1);
        });
        bb.Finalize();
    };

    Array x = Ones({2, 3}, Dtype::kFloat32).RequireGrad();
    Array y1{};
    Array y2{};
    forward(x, y1, y2);
    EXPECT_FALSE(y1.IsBackpropRequired());
    EXPECT_TRUE(y2.IsBackpropRequired());

    Backward(y2);
}

TEST(BackwardBuilderTest, FloatToInt_GetIntRetainOutputFirstParam) {
    testing::ContextSession context_session;
    Shape shape{2, 3};

    auto forward = [shape](const Array& x, Array& y1, Array& y2) {
        y1 = testing::BuildArray(shape).WithData<int32_t>({1, 2, 3, 4, 5, 6});
        y2 = testing::BuildArray(shape).WithData<float>({7, 8, 9, 10, 11, 12});

        BackwardBuilder bb{"forward", x, {y1, y2}};
        BackwardBuilder::Target bt = bb.CreateTarget();
        EXPECT_TRUE(static_cast<bool>(bt));
        bt.Define([out_tok = bb.RetainOutput(0)](BackwardContext& bctx) {
            const absl::optional<Array>& y1 = bctx.GetRetainedOutput(out_tok);
            ASSERT_FALSE(bctx.output_grad(0).has_value());
            ASSERT_TRUE(bctx.output_grad(1).has_value());
            EXPECT_TRUE(y1.has_value());
            bctx.input_grad() = *bctx.output_grad(1) * y1->AsType(Dtype::kFloat32);
        });
        bb.Finalize();
    };

    Array x = Ones(shape, Dtype::kFloat32).RequireGrad();
    Array e = testing::BuildArray(shape).WithData<float>({4, 10, 18, 28, 40, 54});
    Array y1{};
    Array y2{};

    forward(x, y1, y2);
    EXPECT_FALSE(y1.IsBackpropRequired());
    EXPECT_TRUE(y2.IsBackpropRequired());

    y2.SetGrad(testing::BuildArray(shape).WithData<float>({4, 5, 6, 7, 8, 9}));
    Backward(y2);
    EXPECT_TRUE(x.GetGrad().has_value());
    EXPECT_ARRAY_EQ(e, *x.GetGrad());
}

TEST(BackwardBuilderTest, FloatToInt_GetIntRetainOutputSecondParam) {
    testing::ContextSession context_session;
    Shape shape{2, 3};

    auto forward = [shape](const Array& x, Array& y1, Array& y2) {
        y1 = testing::BuildArray(shape).WithData<float>({1, 2, 3, 4, 5, 6});
        y2 = testing::BuildArray(shape).WithData<int32_t>({7, 8, 9, 10, 11, 12});

        BackwardBuilder bb{"forward", x, {y1, y2}};
        BackwardBuilder::Target bt = bb.CreateTarget();
        EXPECT_TRUE(static_cast<bool>(bt));
        bt.Define([out_tok = bb.RetainOutput(1)](BackwardContext& bctx) {
            const absl::optional<Array>& y2 = bctx.GetRetainedOutput(out_tok);
            ASSERT_TRUE(bctx.output_grad(0).has_value());
            ASSERT_FALSE(bctx.output_grad(1).has_value());
            EXPECT_TRUE(y2.has_value());
            bctx.input_grad() = *bctx.output_grad(0) * y2->AsType(Dtype::kFloat32);
        });
        bb.Finalize();
    };

    Array x = Ones(shape, Dtype::kFloat32).RequireGrad();
    Array e = testing::BuildArray(shape).WithData<float>({28, 40, 54, 70, 88, 108});
    Array y1{};
    Array y2{};

    forward(x, y1, y2);
    EXPECT_TRUE(y1.IsBackpropRequired());
    EXPECT_FALSE(y2.IsBackpropRequired());

    y1.SetGrad(testing::BuildArray(shape).WithData<float>({4, 5, 6, 7, 8, 9}));
    Backward(y1);
    EXPECT_TRUE(x.GetGrad().has_value());
    EXPECT_ARRAY_EQ(e, *x.GetGrad());
}

TEST(BackwardBuilderTest, FloatToInt_GetIntRetainOutputArrayBodyIsGone) {
    testing::ContextSession context_session;
    Shape shape{2, 3};

    auto forward = [shape](const Array& x, Array& y1, Array& y2) {
        y1 = testing::BuildArray(shape).WithData<float>({1, 2, 3, 4, 5, 6});
        y2 = testing::BuildArray(shape).WithData<int32_t>({7, 8, 9, 10, 11, 12});

        BackwardBuilder bb{"forward", x, {y1, y2}};
        BackwardBuilder::Target bt = bb.CreateTarget();
        EXPECT_TRUE(static_cast<bool>(bt));
        bt.Define([out_tok = bb.RetainOutput(1)](BackwardContext& bctx) {
            const absl::optional<Array>& y2 = bctx.GetRetainedOutput(out_tok);
            ASSERT_TRUE(bctx.output_grad(0).has_value());
            ASSERT_FALSE(bctx.output_grad(1).has_value());
            EXPECT_TRUE(y2.has_value());
            bctx.input_grad() = *bctx.output_grad(0) * y2->AsType(Dtype::kFloat32);
        });
        bb.Finalize();
    };

    Array x = Ones(shape, Dtype::kFloat32).RequireGrad();
    Array e = testing::BuildArray(shape).WithData<float>({28, 40, 54, 70, 88, 108});
    Array y1{};
    Array z2{};
    {
        Array y2{};

        forward(x, y1, y2);
        EXPECT_TRUE(y1.IsBackpropRequired());
        EXPECT_FALSE(y2.IsBackpropRequired());
        z2 = y2.MakeView();
    }

    y1.SetGrad(testing::BuildArray(shape).WithData<float>({4, 5, 6, 7, 8, 9}));
    Backward(y1);
    EXPECT_TRUE(x.GetGrad().has_value());
    EXPECT_ARRAY_EQ(e, *x.GetGrad());
}

TEST(BackwardBuilderTest, FloatToInt_GetIntRetainOutputArrayNodeIsGone) {
    testing::ContextSession context_session;
    Shape shape{2, 3};

    auto forward = [shape](const Array& x, Array& y1, Array& y2) {
        y1 = testing::BuildArray(shape).WithData<float>({1, 2, 3, 4, 5, 6});
        y2 = testing::BuildArray(shape).WithData<int32_t>({7, 8, 9, 10, 11, 12});

        BackwardBuilder bb{"forward", x, {y1, y2}};
        BackwardBuilder::Target bt = bb.CreateTarget();
        EXPECT_TRUE(static_cast<bool>(bt));
        bt.Define([out_tok = bb.RetainOutput(1)](BackwardContext& bctx) {
            const absl::optional<Array>& y2 = bctx.GetRetainedOutput(out_tok);
            ASSERT_TRUE(bctx.output_grad(0).has_value());
            ASSERT_FALSE(bctx.output_grad(1).has_value());
            EXPECT_TRUE(y2.has_value());
            bctx.input_grad() = *bctx.output_grad(0) * y2->AsType(Dtype::kFloat32);
        });
        bb.Finalize();
    };

    Array x = Ones(shape, Dtype::kFloat32).RequireGrad();
    Array e = testing::BuildArray(shape).WithData<float>({28, 40, 54, 70, 88, 108});
    Array y1{};
    {
        Array y2{};

        forward(x, y1, y2);
        EXPECT_TRUE(y1.IsBackpropRequired());
        EXPECT_FALSE(y2.IsBackpropRequired());
    }

    y1.SetGrad(testing::BuildArray(shape).WithData<float>({4, 5, 6, 7, 8, 9}));
    Backward(y1);
    EXPECT_TRUE(x.GetGrad().has_value());
    EXPECT_ARRAY_EQ(e, *x.GetGrad());
}

}  // namespace
}  // namespace chainerx
