#include "xchainer/squash_dim.h"

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

TEST(SquashDimTest, GetSquashedStrides) {
    EXPECT_EQ(Strides({3, 2, 5, 4}), GetSquashedStrides({3, 2, 5, 4}, {0, 1, 2, 3}));
    EXPECT_EQ(Strides({2, 5}), GetSquashedStrides({3, 2, 5, 4}, {1, 2}));
    EXPECT_EQ(Strides({}), GetSquashedStrides({3, 2, 5, 4}, {}));
    EXPECT_EQ(Strides({}), GetSquashedStrides({}, {}));
}

TEST(SquashDimTest, SquashAllDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>();

    std::tuple<Shape, Axes> squashed_result = SquashShape(a);
    EXPECT_EQ(Shape({shape[0] * shape[1] * shape[2] * shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({3}), std::get<1>(squashed_result));
}

TEST(SquashDimTest, SquashPartialDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});

    std::tuple<Shape, Axes> squashed_result = SquashShape(a);
    EXPECT_EQ(Shape({shape[0] * shape[1], shape[2] * shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({1, 3}), std::get<1>(squashed_result));
}

TEST(SquashDimTest, SquashUnitLengthDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 1, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding(1);

    std::tuple<Shape, Axes> squashed_result = SquashShape(a);
    EXPECT_EQ(Shape({shape[0], shape[1], shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({0, 1, 3}), std::get<1>(squashed_result));
}

TEST(SquashDimTest, SquashMultipleArraysDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});
    Array b = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 0, 1, 0});

    std::tuple<Shape, Axes> squashed_result = SquashShape(a, b);
    EXPECT_EQ(Shape({shape[0] * shape[1], shape[2], shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({1, 2, 3}), std::get<1>(squashed_result));
}

}  // namespace
}  // namespace xchainer
