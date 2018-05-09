#include "xchainer/elementwise.h"

#include <tuple>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

TEST(ElementwiseTest, GetSquashedStrides) {
    {
        Strides squashed = GetSquashedStrides({3, 2, 5, 4}, {0, 1, 2, 3});
        EXPECT_EQ(4, squashed.ndim());
        EXPECT_EQ(3, squashed[0]);
        EXPECT_EQ(2, squashed[1]);
        EXPECT_EQ(5, squashed[2]);
        EXPECT_EQ(4, squashed[3]);
    }
    {
        Strides squashed = GetSquashedStrides({3, 2, 5, 4}, {1, 2});
        EXPECT_EQ(2, squashed.ndim());
        EXPECT_EQ(2, squashed[0]);
        EXPECT_EQ(5, squashed[1]);
    }
    {
        Strides squashed = GetSquashedStrides({3, 2, 5, 4}, {});
        EXPECT_EQ(0, squashed.ndim());
    }
    {
        Strides squashed = GetSquashedStrides({}, {});
        EXPECT_EQ(0, squashed.ndim());
    }
}

TEST(ElementwiseTest, SquashAllDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>();

    Shape squashed{};
    Axes keep{};
    std::tie(squashed, keep) = SquashShape(a);

    EXPECT_EQ(1, squashed.ndim());
    EXPECT_EQ(shape[0] * shape[1] * shape[2] * shape[3], squashed[0]);
    EXPECT_EQ(1, keep.ndim());
    EXPECT_EQ(3, keep[0]);
}

TEST(ElementwiseTest, SquashPartialDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});

    Shape squashed{};
    Axes keep{};
    std::tie(squashed, keep) = SquashShape(a);

    EXPECT_EQ(2, squashed.ndim());
    EXPECT_EQ(shape[0] * shape[1], squashed[0]);
    EXPECT_EQ(shape[2] * shape[3], squashed[1]);
    EXPECT_EQ(2, keep.ndim());
    EXPECT_EQ(1, keep[0]);
    EXPECT_EQ(3, keep[1]);
}

TEST(ElementwiseTest, SquashUnitLengthDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 1, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding(1);

    Shape squashed{};
    Axes keep{};
    std::tie(squashed, keep) = SquashShape(a);

    EXPECT_EQ(3, squashed.ndim());
    EXPECT_EQ(shape[0], squashed[0]);
    EXPECT_EQ(shape[1], squashed[1]);
    EXPECT_EQ(shape[3], squashed[2]);
    EXPECT_EQ(3, keep.ndim());
    EXPECT_EQ(0, keep[0]);
    EXPECT_EQ(1, keep[1]);
    EXPECT_EQ(3, keep[2]);
}

TEST(ElementwiseTest, SquashMultipleArraysDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});
    Array b = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 0, 1, 0});

    Shape squashed{};
    Axes keep{};
    std::tie(squashed, keep) = SquashShape(a, b);

    EXPECT_EQ(3, squashed.ndim());
    EXPECT_EQ(shape[0] * shape[1], squashed[0]);
    EXPECT_EQ(shape[2], squashed[1]);
    EXPECT_EQ(shape[3], squashed[2]);
    EXPECT_EQ(3, keep.ndim());
    EXPECT_EQ(1, keep[0]);
    EXPECT_EQ(2, keep[1]);
    EXPECT_EQ(3, keep[2]);
}

}  // namespace
}  // namespace xchainer
