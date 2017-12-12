#include "xchainer/shape.h"
#include <gtest/gtest.h>

namespace xchainer {
namespace {

void CheckSpanEqual(std::initializer_list<int64_t> expect, gsl::span<const int64_t> actual) {
    EXPECT_EQ(gsl::make_span(expect.begin(), expect.end()), actual);
}

TEST(ShapeTest, Ctor) {
    {
        const Shape shape = {2, 3, 4};
        EXPECT_EQ(3, shape.ndim());
        EXPECT_EQ(static_cast<size_t>(3), shape.size());
        CheckSpanEqual({2, 3, 4}, shape.span());
        EXPECT_EQ(2 * 3 * 4, shape.total_size());
    }
    {
        const std::array<int64_t, 3> dims = {2, 3, 4};
        const Shape shape(gsl::make_span(dims));
        EXPECT_EQ(3, shape.ndim());
        CheckSpanEqual({2, 3, 4}, shape.span());
    }
    {
        const std::array<int64_t, kMaxNdim + 1> too_long = {1};
        EXPECT_THROW(Shape(gsl::make_span(too_long)), DimensionError);
    }
}

TEST(ShapeTest, Copy) {
    const Shape shape = {2, 3, 4};
    const Shape shape2 = shape;
    EXPECT_EQ(shape.ndim(), shape2.ndim());
    CheckSpanEqual({2, 3, 4}, shape.span());
}

TEST(ShapeTest, Move) {
    Shape shape = {2, 3, 4};
    Shape shape2 = std::move(shape);
    EXPECT_EQ(shape.ndim(), shape2.ndim());
    CheckSpanEqual({2, 3, 4}, shape.span());
}

}  // namespace
}  // namespace xchainer
