#include "chainerx/optional_container_arg.h"

#include <algorithm>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

namespace chainerx {
namespace {

template <typename T>
const T& AsConst(const T& value) {
    return value;
}

// Initialize with null
TEST(OptionalContainerArgTest, Null) {
    OptionalContainerArg<std::vector<int>> a{absl::nullopt};
    EXPECT_FALSE(a.has_value());
    EXPECT_FALSE(static_cast<bool>(a));
}

// Initialize with empty vector
TEST(OptionalContainerArgTest, EmptyVector) {
    {
        OptionalContainerArg<std::vector<int>> a{};
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(size_t{0}, a->size());
    }
    {
        OptionalContainerArg<std::vector<int>> a{std::vector<int>{}};
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(size_t{0}, a->size());
    }
}

// Initialize with vector
TEST(OptionalContainerArgTest, Vector) {
    {
        OptionalContainerArg<std::vector<int>> a{{5, -2, 3}};
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(size_t{3}, a->size());
        EXPECT_EQ(std::vector<int>({5, -2, 3}), *a);

        const std::vector<int>& vec = *a;
        EXPECT_EQ(&vec, &*a);  // vec is not copied
    }
    {
        OptionalContainerArg<std::vector<int>> a{std::vector<int>{5, -2, 3}};
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(size_t{3}, a->size());
        EXPECT_EQ(std::vector<int>({5, -2, 3}), *a);
    }
}

// Initialize with single-element vector
TEST(OptionalContainerArgTest, SingleVector) {
    {
        OptionalContainerArg<std::vector<int>> a{0};
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(std::vector<int>({0}), *a);
    }

    {
        OptionalContainerArg<std::vector<int>> a{std::vector<int>{1}};
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(std::vector<int>({1}), *a);
    }
}

// Initialize with single value
TEST(OptionalContainerArgTest, Single) {
    {
        OptionalContainerArg<std::vector<int>> a = 0;
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(std::vector<int>{0}, *a);
    }

    {
        OptionalContainerArg<std::vector<int>> a = 1;
        EXPECT_TRUE(a.has_value());
        EXPECT_TRUE(static_cast<bool>(a));
        EXPECT_EQ(std::vector<int>{1}, *a);
    }
}

// Assign null
TEST(OptionalContainerArgTest, AssignNull) {
    OptionalContainerArg<std::vector<int>> a{5, -2, 3};
    a = absl::nullopt;
    EXPECT_FALSE(a.has_value());
    EXPECT_FALSE(static_cast<bool>(a));
}

// Assign single value
TEST(OptionalContainerArgTest, AssignSingle) {
    OptionalContainerArg<std::vector<int>> a{5, -2, 3};
    a = 6;
    EXPECT_TRUE(a.has_value());
    EXPECT_TRUE(static_cast<bool>(a));
    EXPECT_EQ(std::vector<int>({6}), *a);
}

// Assign vector
TEST(OptionalContainerArgTest, AssignVector) {
    OptionalContainerArg<std::vector<int>> a{5, -2, 3};
    a = std::vector<int>{8, 4, 7};
    EXPECT_TRUE(a.has_value());
    EXPECT_TRUE(static_cast<bool>(a));
    EXPECT_EQ(std::vector<int>({8, 4, 7}), *a);
}

// Copy ctor
TEST(OptionalContainerArgTest, CopyCtor) {
    OptionalContainerArg<std::vector<int>> a{5, -2, 3};
    OptionalContainerArg<std::vector<int>> b = a;
    EXPECT_EQ(std::vector<int>({5, -2, 3}), *b);
    EXPECT_NE(&*a, &*b);
}

// Move ctor
TEST(OptionalContainerArgTest, MoveCtor) {
    OptionalContainerArg<std::vector<int>> a{5, -2, 3};
    int* data_ptr = a->data();
    OptionalContainerArg<std::vector<int>> b = std::move(a);
    EXPECT_EQ(std::vector<int>({5, -2, 3}), *b);
    EXPECT_EQ(data_ptr, b->data());
}

// value()
TEST(OptionalContainerArgTest, Value) {
    OptionalContainerArg<std::vector<int>> a{{5, -2, 3}};
    EXPECT_EQ(std::vector<int>({5, -2, 3}), a.value());
    EXPECT_EQ(std::vector<int>({5, -2, 3}), AsConst(a).value());

    // assign via value()
    a.value() = std::vector<int>({4});
    EXPECT_EQ(std::vector<int>({4}), *a);

    // value() for unset value
    a = absl::nullopt;
    EXPECT_THROW(a.value(), absl::bad_optional_access);
    EXPECT_THROW(AsConst(a).value(), absl::bad_optional_access);
}

// Modify using iterators
TEST(OptionalContainerArgTest, ModifyIterators) {
    OptionalContainerArg<std::vector<int>> a{{5, -2, 3}};
    std::sort(a->begin(), a->end());
    EXPECT_EQ(std::vector<int>({-2, 3, 5}), *a);
}

// Modify using operator*
TEST(OptionalContainerArgTest, ModifyVectorAssign) {
    OptionalContainerArg<std::vector<int>> a{{5, -2, 3}};
    *a = std::vector<int>{9, 8};
    EXPECT_EQ(std::vector<int>({9, 8}), *a);
}

}  // namespace
}  // namespace chainerx
