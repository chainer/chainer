#include "chainerx/backward_builder.h"

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/backward.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"
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
            bctx.input_grad() = bctx.output_grad(0) + bctx.output_grad(1);
        });
        bb.Finalize();
    };

    Array x = Ones({2, 3}, Dtype::kFloat32).RequireGrad();
    Array y1{};
    Array y2{};
    forward(x, y1, y2);
    EXPECT_FALSE(y1.IsBackpropRequired());
    EXPECT_TRUE(y2.IsBackpropRequired());

    // TODO(niboshi): Currently this test does not pass with the following line. Fix it.
    // Backward(y2);
}

}  // namespace
}  // namespace chainerx
