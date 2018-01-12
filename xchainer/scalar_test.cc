#include "xchainer/scalar.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace {

TEST(ScalarTest, Dtype) {
    EXPECT_EQ(Scalar(true).dtype(), Dtype::kBool);
    EXPECT_EQ(Scalar(false).dtype(), Dtype::kBool);
    EXPECT_EQ(Scalar(int8_t(1)).dtype(), Dtype::kInt8);
    EXPECT_EQ(Scalar(int16_t(2)).dtype(), Dtype::kInt16);
    EXPECT_EQ(Scalar(int32_t(3)).dtype(), Dtype::kInt32);
    EXPECT_EQ(Scalar(int64_t(4)).dtype(), Dtype::kInt64);
    EXPECT_EQ(Scalar(uint8_t(5)).dtype(), Dtype::kUInt8);
    EXPECT_EQ(Scalar(6.7f).dtype(), Dtype::kFloat32);
    EXPECT_EQ(Scalar(8.9).dtype(), Dtype::kFloat64);
}

TEST(ScalarTest, Cast) {
    EXPECT_TRUE(bool(Scalar(true)));
    EXPECT_TRUE(bool(Scalar(1)));
    EXPECT_TRUE(bool(Scalar(-3.2)));

    EXPECT_FALSE(bool(Scalar(false)));
    EXPECT_FALSE(bool(Scalar(0)));
    EXPECT_FALSE(bool(Scalar(0.0f)));

    EXPECT_EQ(int8_t(Scalar(1)), 1);
    EXPECT_EQ(int8_t(Scalar(-1.1f)), -1);
    EXPECT_EQ(int8_t(Scalar(1.1)), 1);

    EXPECT_EQ(int16_t(Scalar(-2)), -2);
    EXPECT_EQ(int16_t(Scalar(2.2f)), 2);
    EXPECT_EQ(int16_t(Scalar(2.2)), 2);

    EXPECT_EQ(int32_t(Scalar(3)), 3);
    EXPECT_EQ(int32_t(Scalar(3.3f)), 3);
    EXPECT_EQ(int32_t(Scalar(-3.3)), -3);

    EXPECT_EQ(int64_t(Scalar(4)), 4);
    EXPECT_EQ(int64_t(Scalar(4.4f)), 4);
    EXPECT_EQ(int64_t(Scalar(-4.4)), -4);

    EXPECT_EQ(uint8_t(Scalar(5)), 5);
    EXPECT_EQ(uint8_t(Scalar(5.5f)), 5);
    EXPECT_EQ(uint8_t(Scalar(5.0)), 5);

    EXPECT_FLOAT_EQ(float(Scalar(-6)), -6.0f);
    EXPECT_FLOAT_EQ(float(Scalar(6.7f)), 6.7f);
    EXPECT_FLOAT_EQ(float(Scalar(6.7)), 6.7f);

    EXPECT_DOUBLE_EQ(double(Scalar(8)), 8.0);
    EXPECT_DOUBLE_EQ(double(Scalar(-8.9f)), double(-8.9f));
    EXPECT_DOUBLE_EQ(double(Scalar(8.9)), 8.9);
}

TEST(DtypeTest, UnaryOps) {
    EXPECT_THROW(-Scalar(true), DtypeError);
    EXPECT_THROW(-Scalar(false), DtypeError);
    EXPECT_EQ(int8_t(-Scalar(1)), -1);
    EXPECT_EQ(int16_t(-Scalar(2)), -2);
    EXPECT_EQ(int32_t(-Scalar(3)), -3);
    EXPECT_EQ(int64_t(-Scalar(4)), -4);
    EXPECT_EQ(uint8_t(-Scalar(5)), uint8_t(-5));
    EXPECT_FLOAT_EQ(float(-Scalar(6)), -6.0f);
    EXPECT_FLOAT_EQ(float(-Scalar(6.7)), -6.7f);
    EXPECT_DOUBLE_EQ(double(-Scalar(8)), -8.0);
    EXPECT_DOUBLE_EQ(double(-Scalar(8.9)), -8.9);

    EXPECT_EQ(int8_t(+Scalar(1)), 1);
    EXPECT_EQ(int16_t(+Scalar(2)), 2);
    EXPECT_EQ(int32_t(+Scalar(3)), 3);
    EXPECT_EQ(int64_t(+Scalar(4)), 4);
    EXPECT_EQ(uint8_t(+Scalar(5)), 5);
    EXPECT_EQ(float(+Scalar(5)), 5);
    EXPECT_FLOAT_EQ(float(+Scalar(6)), 6.0f);
    EXPECT_FLOAT_EQ(float(+Scalar(6.7)), 6.7f);
    EXPECT_DOUBLE_EQ(double(+Scalar(8)), 8.0);
    EXPECT_DOUBLE_EQ(double(+Scalar(8.9)), 8.9);
}

TEST(ScalarTest, ToString) {
    EXPECT_EQ(Scalar(true).ToString(), "True");
    EXPECT_EQ(Scalar(false).ToString(), "False");
    EXPECT_EQ(Scalar(int8_t(1)).ToString(), std::to_string(int8_t(1)));
    EXPECT_EQ(Scalar(int16_t(2)).ToString(), std::to_string(int16_t(2)));
    EXPECT_EQ(Scalar(int32_t(3)).ToString(), std::to_string(int32_t(3)));
    EXPECT_EQ(Scalar(int64_t(4)).ToString(), std::to_string(int64_t(4)));
    EXPECT_EQ(Scalar(uint8_t(5)).ToString(), std::to_string(uint8_t(5)));
}

}  // namespace
}  // namespace xchainer
