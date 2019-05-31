#include "chainerx/scalar.h"

#include <gtest/gtest.h>

namespace chainerx {
namespace {

TEST(ScalarTest, Type) {
    EXPECT_EQ(Scalar(true).kind(), DtypeKind::kBool);
    EXPECT_EQ(Scalar(false).kind(), DtypeKind::kBool);
    EXPECT_EQ(Scalar(int8_t(1)).kind(), DtypeKind::kInt);
    EXPECT_EQ(Scalar(int16_t(2)).kind(), DtypeKind::kInt);
    EXPECT_EQ(Scalar(int32_t(3)).kind(), DtypeKind::kInt);
    EXPECT_EQ(Scalar(int64_t(4)).kind(), DtypeKind::kInt);
    EXPECT_EQ(Scalar(uint8_t(5)).kind(), DtypeKind::kInt);
    EXPECT_EQ(Scalar(6.7f).kind(), DtypeKind::kFloat);
    EXPECT_EQ(Scalar(8.9).kind(), DtypeKind::kFloat);
}

template <typename T1, typename T2>
void ExpectScalarEqual(T1 value1, T2 value2) {
    EXPECT_EQ(Scalar(value1), Scalar(value2));
    EXPECT_EQ(Scalar(value2), Scalar(value1));
}

TEST(ScalarTest, Equality) {
    // Same primitive type
    ExpectScalarEqual(int8_t{0}, int8_t{0});
    ExpectScalarEqual(int16_t{0}, int16_t{0});
    ExpectScalarEqual(int32_t{0}, int32_t{0});
    ExpectScalarEqual(int64_t{0}, int64_t{0});
    ExpectScalarEqual(uint8_t{0}, uint8_t{0});
    ExpectScalarEqual(uint16_t{0}, uint16_t{0});
    ExpectScalarEqual(uint32_t{0}, uint32_t{0});
    ExpectScalarEqual(int8_t{1}, int8_t{1});
    ExpectScalarEqual(int16_t{1}, int16_t{1});
    ExpectScalarEqual(int32_t{1}, int32_t{1});
    ExpectScalarEqual(int64_t{1}, int64_t{1});
    ExpectScalarEqual(uint8_t{1}, uint8_t{1});
    ExpectScalarEqual(uint16_t{1}, uint16_t{1});
    ExpectScalarEqual(uint32_t{1}, uint32_t{1});
    ExpectScalarEqual(1.5, 1.5);
    ExpectScalarEqual(1.5f, 1.5f);
    ExpectScalarEqual(-1.5, -1.5);
    ExpectScalarEqual(-1.5f, -1.5f);
    ExpectScalarEqual(true, true);
    ExpectScalarEqual(false, false);

    // Different primitive types and same kind
    ExpectScalarEqual(uint8_t{1}, int64_t{1});
    ExpectScalarEqual(uint8_t{1}, uint32_t{1});
    ExpectScalarEqual(int8_t{1}, int32_t{1});
    ExpectScalarEqual(1.5f, 1.5);

    // Different primitive types and different kinds
    ExpectScalarEqual(int32_t{1}, 1.0f);
    ExpectScalarEqual(true, int16_t{1});
    ExpectScalarEqual(false, int16_t{0});
    ExpectScalarEqual(false, 0.0f);
    ExpectScalarEqual(true, 1.0f);
}

template <typename T1, typename T2>
void ExpectScalarNotEqual(T1 value1, T2 value2) {
    EXPECT_NE(Scalar(value1), Scalar(value2));
    EXPECT_NE(Scalar(value2), Scalar(value1));
}

TEST(ScalarTest, Inequality) {
    // Same primitive type
    ExpectScalarNotEqual(0, 1);
    ExpectScalarNotEqual(-1, 1);
    ExpectScalarNotEqual(-1.0001, -1.0);
    ExpectScalarNotEqual(-1.0001, -1);
    ExpectScalarNotEqual(true, false);
    ExpectScalarNotEqual(true, 1.1);
    ExpectScalarNotEqual(1.0001, 1.0002);
    ExpectScalarNotEqual(std::nan(""), std::nan(""));

    // Different primitive types and same kind
    ExpectScalarNotEqual(int32_t{1}, int16_t{2});
    ExpectScalarNotEqual(uint8_t{1}, int8_t{2});
    ExpectScalarNotEqual(uint16_t{1}, uint8_t{2});
    ExpectScalarNotEqual(uint8_t{0xff}, int8_t{-1});
    ExpectScalarNotEqual(1.0f, 2.0);

    // Different primitive types and different kinds
    ExpectScalarNotEqual(int32_t{2}, 1.0);
    ExpectScalarNotEqual(true, int16_t{2});
    ExpectScalarNotEqual(true, int16_t{-1});
    ExpectScalarNotEqual(false, int16_t{2});
    ExpectScalarNotEqual(false, int16_t{-1});
    ExpectScalarNotEqual(false, 0.1);
    ExpectScalarNotEqual(true, 0.9);
    ExpectScalarNotEqual(true, -1);
    ExpectScalarNotEqual(true, std::nan(""));
    ExpectScalarNotEqual(false, std::nan(""));
}

TEST(ScalarTest, Cast) {
    EXPECT_TRUE(static_cast<bool>(Scalar(true)));
    EXPECT_TRUE(static_cast<bool>(Scalar(1)));
    EXPECT_TRUE(static_cast<bool>(Scalar(-3.2)));

    EXPECT_FALSE(static_cast<bool>(Scalar(false)));
    EXPECT_FALSE(static_cast<bool>(Scalar(0)));
    EXPECT_FALSE(static_cast<bool>(Scalar(0.0f)));

    EXPECT_EQ(static_cast<int8_t>(Scalar(1)), 1);
    EXPECT_EQ(static_cast<int8_t>(Scalar(-1.1f)), -1);
    EXPECT_EQ(static_cast<int8_t>(Scalar(1.1)), 1);

    EXPECT_EQ(static_cast<int16_t>(Scalar(-2)), -2);
    EXPECT_EQ(static_cast<int16_t>(Scalar(2.2f)), 2);
    EXPECT_EQ(static_cast<int16_t>(Scalar(2.2)), 2);

    EXPECT_EQ(static_cast<int32_t>(Scalar(3)), 3);
    EXPECT_EQ(static_cast<int32_t>(Scalar(3.3f)), 3);
    EXPECT_EQ(static_cast<int32_t>(Scalar(-3.3)), -3);

    EXPECT_EQ(static_cast<int64_t>(Scalar(4)), 4);
    EXPECT_EQ(static_cast<int64_t>(Scalar(4.4f)), 4);
    EXPECT_EQ(static_cast<int64_t>(Scalar(-4.4)), -4);

    EXPECT_EQ(static_cast<uint8_t>(Scalar(5)), 5);
    EXPECT_EQ(static_cast<uint8_t>(Scalar(5.5f)), 5);
    EXPECT_EQ(static_cast<uint8_t>(Scalar(5.0)), 5);

    EXPECT_FLOAT_EQ(static_cast<float>(Scalar(-6)), -6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(Scalar(6.7f)), 6.7f);
    EXPECT_FLOAT_EQ(static_cast<float>(Scalar(6.7)), 6.7f);

    EXPECT_DOUBLE_EQ(static_cast<double>(Scalar(8)), 8.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(Scalar(-8.9f)), double{-8.9f});
    EXPECT_DOUBLE_EQ(static_cast<double>(Scalar(8.9)), 8.9);
}

TEST(DtypeTest, UnaryOps) {
    EXPECT_THROW(-Scalar(true), DtypeError);
    EXPECT_THROW(-Scalar(false), DtypeError);
    EXPECT_EQ(static_cast<int8_t>(-Scalar(1)), -1);
    EXPECT_EQ(static_cast<int16_t>(-Scalar(2)), -2);
    EXPECT_EQ(static_cast<int32_t>(-Scalar(3)), -3);
    EXPECT_EQ(static_cast<int64_t>(-Scalar(4)), -4);
    EXPECT_EQ(static_cast<uint8_t>(-Scalar(5)), static_cast<uint8_t>(-5));
    EXPECT_FLOAT_EQ(static_cast<float>(-Scalar(6)), -6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(-Scalar(6.7)), -6.7f);
    EXPECT_DOUBLE_EQ(static_cast<double>(-Scalar(8)), -8.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(-Scalar(8.9)), -8.9);

    EXPECT_EQ(static_cast<int8_t>(+Scalar(1)), 1);
    EXPECT_EQ(static_cast<int16_t>(+Scalar(2)), 2);
    EXPECT_EQ(static_cast<int32_t>(+Scalar(3)), 3);
    EXPECT_EQ(static_cast<int64_t>(+Scalar(4)), 4);
    EXPECT_EQ(static_cast<uint8_t>(+Scalar(5)), 5);
    EXPECT_EQ(static_cast<float>(+Scalar(5)), 5);
    EXPECT_FLOAT_EQ(static_cast<float>(+Scalar(6)), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(+Scalar(6.7)), 6.7f);
    EXPECT_DOUBLE_EQ(static_cast<double>(+Scalar(8)), 8.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(+Scalar(8.9)), 8.9);
}

TEST(ScalarTest, ToString) {
    EXPECT_EQ(Scalar(true).ToString(), "True");
    EXPECT_EQ(Scalar(false).ToString(), "False");
    EXPECT_EQ(Scalar(int8_t{1}).ToString(), std::to_string(int8_t{1}));
    EXPECT_EQ(Scalar(int16_t{2}).ToString(), std::to_string(int16_t{2}));
    EXPECT_EQ(Scalar(int32_t{3}).ToString(), std::to_string(int32_t{3}));
    EXPECT_EQ(Scalar(int64_t{4}).ToString(), std::to_string(int64_t{4}));
    EXPECT_EQ(Scalar(uint8_t{5}).ToString(), std::to_string(uint8_t{5}));
}

}  // namespace
}  // namespace chainerx
