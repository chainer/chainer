#include "xchainer/shape.h"
#include <gtest/gtest.h>

namespace xchainer {
namespace {

void CheckSpanEqual(std::initializer_list<int64_t> expect, gsl::span<const int64_t> actual) {
    ASSERT_EQ(gsl::make_span(expect.begin(), expect.end()), actual);
}

TEST(ShapeTest, Ctor) {
    {
        const Shape shape = {2, 3, 4};
        ASSERT_EQ(3, shape.ndim());
        ASSERT_EQ(static_cast<size_t>(3), shape.size());
        CheckSpanEqual({2, 3, 4}, shape.span());
        ASSERT_EQ(2 * 3 * 4, shape.total_size());
    }
    {
        const std::array<int64_t, 3> dims = {2, 3, 4};
        const Shape shape(gsl::make_span(dims));
        ASSERT_EQ(3, shape.ndim());
        CheckSpanEqual({2, 3, 4}, shape.span());
    }
    {
        const std::array<int64_t, kMaxNdim + 1> too_long = {1};
        ASSERT_THROW(Shape(gsl::make_span(too_long)), DimensionError);
    }
}

TEST(ShapeTest, Subscript) {
    const Shape shape = {2, 3, 4};
    ASSERT_EQ(2, shape[0]);
    ASSERT_EQ(3, shape[1]);
    ASSERT_EQ(4, shape[2]);
    ASSERT_THROW(shape[-1], DimensionError);
    ASSERT_THROW(shape[3], DimensionError);
}

TEST(ShapeTest, Compare) {
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {2, 3, 4};
        ASSERT_TRUE(shape == shape2);
    }
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {2, 3};
        ASSERT_TRUE(shape != shape2);
    }
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {1, 2, 3};
        ASSERT_TRUE(shape != shape2);
    }
}

TEST(ShapeTest, CheckEqual) {
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {2, 3, 4};
        ASSERT_NO_THROW(CheckEqual(shape, shape2));
    }
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {};
        ASSERT_THROW(CheckEqual(shape, shape2), DimensionError);
    }
}

}  // namespace
}  // namespace xchainer
