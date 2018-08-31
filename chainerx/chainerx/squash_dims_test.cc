#include "chainerx/squash_dims.h"

#include <tuple>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/context_session.h"

namespace chainerx {
namespace {

TEST(SquashDimsTest, GetSquashedStrides) {
    EXPECT_EQ(Strides({3, 2, 5, 4}), GetSquashedStrides({3, 2, 5, 4}, {0, 1, 2, 3}));
    EXPECT_EQ(Strides({2, 5}), GetSquashedStrides({3, 2, 5, 4}, {1, 2}));
    EXPECT_EQ(Strides({}), GetSquashedStrides({3, 2, 5, 4}, {}));
    EXPECT_EQ(Strides({}), GetSquashedStrides({}, {}));
}

TEST(SquashDimsTest, SquashAllDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>();

    std::tuple<Shape, Axes> squashed_result = SquashShape(a);
    EXPECT_EQ(Shape({shape[0] * shape[1] * shape[2] * shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({3}), std::get<1>(squashed_result));
}

TEST(SquashDimsTest, SquashPartialDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});

    std::tuple<Shape, Axes> squashed_result = SquashShape(a);
    EXPECT_EQ(Shape({shape[0] * shape[1], shape[2] * shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({1, 3}), std::get<1>(squashed_result));
}

TEST(SquashDimsTest, SquashUnitLengthDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 1, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding(1);

    std::tuple<Shape, Axes> squashed_result = SquashShape(a);
    EXPECT_EQ(Shape({shape[0], shape[1], shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({0, 1, 3}), std::get<1>(squashed_result));
}

TEST(SquashDimsTest, SquashMultipleArraysDimensions) {
    testing::ContextSession context_session;
    Shape shape{3, 2, 5, 4};
    Array a = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 2, 0, 0});
    Array b = testing::BuildArray(shape).WithLinearData<float>().WithPadding({0, 0, 1, 0});

    std::tuple<Shape, Axes> squashed_result = SquashShape(a, b);
    EXPECT_EQ(Shape({shape[0] * shape[1], shape[2], shape[3]}), std::get<0>(squashed_result));
    EXPECT_EQ(Axes({1, 2, 3}), std::get<1>(squashed_result));
}

}  // namespace
}  // namespace chainerx
