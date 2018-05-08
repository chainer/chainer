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

TEST(ElementwiseTest, Reduce) {
    {
        Strides reduced = Reduce({3, 2, 5, 4}, {0, 1, 2, 3});
        EXPECT_EQ(4, reduced.ndim());
        EXPECT_EQ(3, reduced[0]);
        EXPECT_EQ(2, reduced[1]);
        EXPECT_EQ(5, reduced[2]);
        EXPECT_EQ(4, reduced[3]);
    }
    {
        Strides reduced = Reduce({3, 2, 5, 4}, {1, 2});
        EXPECT_EQ(2, reduced.ndim());
        EXPECT_EQ(2, reduced[0]);
        EXPECT_EQ(5, reduced[1]);
    }
    {
        Strides reduced = Reduce({3, 2, 5, 4}, {});
        EXPECT_EQ(0, reduced.ndim());
    }
    {
        Strides reduced = Reduce({}, {});
        EXPECT_EQ(0, reduced.ndim());
    }
}

TEST(ElementwiseTest, ReduceAllDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>();

    Shape reduced{};
    Axes keep{};
    std::tie(reduced, keep) = ReducedShape(a);

    EXPECT_EQ(1, reduced.ndim());
    EXPECT_EQ(shape[0] * shape[1] * shape[2] * shape[3], reduced[0]);
    EXPECT_EQ(1, keep.ndim());
    EXPECT_EQ(3, keep[0]);
}

TEST(ElementwiseTest, ReducePartialDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});

    Shape reduced{};
    Axes keep{};
    std::tie(reduced, keep) = ReducedShape(a);

    EXPECT_EQ(2, reduced.ndim());
    EXPECT_EQ(shape[0] * shape[1], reduced[0]);
    EXPECT_EQ(shape[2] * shape[3], reduced[1]);
    EXPECT_EQ(2, keep.ndim());
    EXPECT_EQ(1, keep[0]);
    EXPECT_EQ(3, keep[1]);
}

TEST(ElementwiseTest, ReduceUnitLengthDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 1, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding(1);

    Shape reduced{};
    Axes keep{};
    std::tie(reduced, keep) = ReducedShape(a);

    EXPECT_EQ(3, reduced.ndim());
    EXPECT_EQ(shape[0], reduced[0]);
    EXPECT_EQ(shape[1], reduced[1]);
    EXPECT_EQ(shape[3], reduced[2]);
    EXPECT_EQ(3, keep.ndim());
    EXPECT_EQ(0, keep[0]);
    EXPECT_EQ(1, keep[1]);
    EXPECT_EQ(3, keep[2]);
}

TEST(ElementwiseTest, ReduceMultipleArraysDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});
    Array b = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 0, 1, 0});

    Shape reduced{};
    Axes keep{};
    std::tie(reduced, keep) = ReducedShape(a, b);

    EXPECT_EQ(3, reduced.ndim());
    EXPECT_EQ(shape[0] * shape[1], reduced[0]);
    EXPECT_EQ(shape[2], reduced[1]);
    EXPECT_EQ(shape[3], reduced[2]);
    EXPECT_EQ(3, keep.ndim());
    EXPECT_EQ(1, keep[0]);
    EXPECT_EQ(2, keep[1]);
    EXPECT_EQ(3, keep[2]);
}

}  // namespace
}  // namespace xchainer
