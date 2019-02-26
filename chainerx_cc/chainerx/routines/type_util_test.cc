#include "chainerx/routines/type_util.h"

#include <vector>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"
#include "chainerx/testing/context_session.h"

namespace chainerx {
namespace {

// Declares necessary variables and runs a set of checks.
#define CHECK_RESULT_TYPE_IMPL(check_body)           \
    {                                                \
        testing::ContextSession context_session;     \
        Array Ab = Empty({2, 3}, Dtype::kBool);      \
        Array Au8 = Empty({2, 3}, Dtype::kUInt8);    \
        Array Ai8 = Empty({2, 3}, Dtype::kInt8);     \
        Array Ai16 = Empty({2, 3}, Dtype::kInt16);   \
        Array Ai32 = Empty({2, 3}, Dtype::kInt32);   \
        Array Ai64 = Empty({2, 3}, Dtype::kInt64);   \
        Array Af16 = Empty({2, 3}, Dtype::kFloat16); \
        Array Af32 = Empty({2, 3}, Dtype::kFloat32); \
        Array Af64 = Empty({2, 3}, Dtype::kFloat64); \
        Scalar Sb{1, Dtype::kBool};                  \
        Scalar Su8{1, Dtype::kUInt8};                \
        Scalar Si8{1, Dtype::kInt8};                 \
        Scalar Si16{1, Dtype::kInt16};               \
        Scalar Si32{1, Dtype::kInt32};               \
        Scalar Si64{1, Dtype::kInt64};               \
        Scalar Sf16{1, Dtype::kFloat16};             \
        Scalar Sf32{1, Dtype::kFloat32};             \
        Scalar Sf64{1, Dtype::kFloat64};             \
        check_body;                                  \
    }

// Checks ResultType: static-length 1-arg call
// args are Ai8, Sf32, etc.
#define CHECK_RESULT_TYPE1(arg1) CHECK_RESULT_TYPE_IMPL({ EXPECT_EQ((arg1).dtype(), chainerx::ResultType(arg1)); })

// Checks ResultType: static-length 2-arg call
// args are Ai8, Sf32, etc.
#define CHECK_RESULT_TYPE2(expected_dtype, arg1, arg2)                         \
    CHECK_RESULT_TYPE_IMPL({                                                   \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg1, arg2)); \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg2, arg1)); \
    })

// Checks ResultType: static-length 3-arg call
// args are Ai8, Sf32, etc.
#define CHECK_RESULT_TYPE3(expected_dtype, arg1, arg2, arg3)                         \
    CHECK_RESULT_TYPE_IMPL({                                                         \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg1, arg2, arg3)); \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg1, arg3, arg2)); \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg2, arg1, arg3)); \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg2, arg3, arg1)); \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg3, arg1, arg2)); \
        EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(arg3, arg2, arg1)); \
    })

// Checks ResultType: 2 homogeneous-type arguments
// hargs are i8, f32, etc.
#define CHECK_RESULT_TYPE_HOMO1(harg1)                                                                                \
    {                                                                                                                 \
        CHECK_RESULT_TYPE1(A##harg1);                                                                                 \
        CHECK_RESULT_TYPE1(S##harg1);                                                                                 \
        /* Dynamic-length arrays */                                                                                   \
        CHECK_RESULT_TYPE_IMPL({ EXPECT_EQ(A##harg1.dtype(), chainerx::ResultType(std::vector<Array>{A##harg1})); }); \
    }

// Checks ResultType: 2 homogeneous-type arguments
// hargs are i8, f32, etc.
#define CHECK_RESULT_TYPE_HOMO2(expected_dtype, harg1, harg2)                                                  \
    {                                                                                                          \
        CHECK_RESULT_TYPE2(expected_dtype, A##harg1, A##harg2);                                                \
        CHECK_RESULT_TYPE2(expected_dtype, S##harg1, S##harg2);                                                \
        /* Dynamic-length arrays */                                                                            \
        CHECK_RESULT_TYPE_IMPL({                                                                               \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg1, A##harg2})); \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg2, A##harg1})); \
        });                                                                                                    \
    }

// Checks ResultType: 3 homogeneous-type arguments
// hargs are i8, f32, etc.
#define CHECK_RESULT_TYPE_HOMO3(expected_dtype, harg1, harg2, harg3)                                                     \
    {                                                                                                                    \
        CHECK_RESULT_TYPE3(expected_dtype, A##harg1, A##harg2, A##harg3);                                                \
        CHECK_RESULT_TYPE3(expected_dtype, S##harg1, S##harg2, S##harg3);                                                \
        /* Dynamic-length arrays */                                                                                      \
        CHECK_RESULT_TYPE_IMPL({                                                                                         \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg1, A##harg2, A##harg3})); \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg1, A##harg3, A##harg2})); \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg2, A##harg1, A##harg3})); \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg2, A##harg3, A##harg1})); \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg3, A##harg1, A##harg2})); \
            EXPECT_EQ(Dtype::k##expected_dtype, chainerx::ResultType(std::vector<Array>{A##harg3, A##harg2, A##harg1})); \
        });                                                                                                              \
    }

TEST(ResultTypeTest, NoArgs) {
    EXPECT_THROW({ chainerx::ResultType(std::vector<Array>{}); }, ChainerxError);
}

TEST(ResultTypeTest, One) {
    CHECK_RESULT_TYPE_HOMO1(b);
    CHECK_RESULT_TYPE_HOMO1(f16);
    CHECK_RESULT_TYPE_HOMO1(f32);
    CHECK_RESULT_TYPE_HOMO1(f64);
    CHECK_RESULT_TYPE_HOMO1(i8);
    CHECK_RESULT_TYPE_HOMO1(i16);
    CHECK_RESULT_TYPE_HOMO1(i32);
    CHECK_RESULT_TYPE_HOMO1(i64);
    CHECK_RESULT_TYPE_HOMO1(u8);
}

TEST(ResultTypeTest, TwoBool) { CHECK_RESULT_TYPE_HOMO2(Bool, b, b); }

TEST(ResultTypeTest, TwoFloat) {
    CHECK_RESULT_TYPE_HOMO2(Float16, f16, f16);
    CHECK_RESULT_TYPE_HOMO2(Float32, f32, f32);
    CHECK_RESULT_TYPE_HOMO2(Float64, f64, f64);
    CHECK_RESULT_TYPE_HOMO2(Float32, f16, f32);
    CHECK_RESULT_TYPE_HOMO2(Float64, f16, f64);
    CHECK_RESULT_TYPE_HOMO2(Float64, f32, f64);
}

TEST(ResultTypeTest, TwoSignedInt) {
    CHECK_RESULT_TYPE_HOMO2(Int8, i8, i8);
    CHECK_RESULT_TYPE_HOMO2(Int16, i8, i16);
    CHECK_RESULT_TYPE_HOMO2(Int32, i8, i32);
    CHECK_RESULT_TYPE_HOMO2(Int64, i8, i64);
    CHECK_RESULT_TYPE_HOMO2(Int16, i16, i16);
    CHECK_RESULT_TYPE_HOMO2(Int32, i32, i32);
    CHECK_RESULT_TYPE_HOMO2(Int64, i64, i64);
    CHECK_RESULT_TYPE_HOMO2(Int32, i16, i32);
    CHECK_RESULT_TYPE_HOMO2(Int64, i16, i64);
    CHECK_RESULT_TYPE_HOMO2(Int64, i32, i64);
}

TEST(ResultTypeTest, TwoUnsignedInt) { CHECK_RESULT_TYPE_HOMO2(UInt8, u8, u8); }

TEST(ResultTypeTest, TwoSignedIntAndUnsignedInt) {
    CHECK_RESULT_TYPE_HOMO2(Int16, u8, i8);
    CHECK_RESULT_TYPE_HOMO2(Int16, u8, i16);
    CHECK_RESULT_TYPE_HOMO2(Int32, u8, i32);
}

TEST(ResultTypeTest, TwoIntAndFloat) {
    CHECK_RESULT_TYPE_HOMO2(Float16, i8, f16);
    CHECK_RESULT_TYPE_HOMO2(Float16, u8, f16);
    CHECK_RESULT_TYPE_HOMO2(Float32, i16, f32);
    CHECK_RESULT_TYPE_HOMO2(Float32, i32, f32);
    CHECK_RESULT_TYPE_HOMO2(Float32, i64, f32);
}

TEST(ResultTypeTest, TwoBoolAndOther) {
    CHECK_RESULT_TYPE_HOMO2(UInt8, b, u8);
    CHECK_RESULT_TYPE_HOMO2(Int8, b, i8);
    CHECK_RESULT_TYPE_HOMO2(Int16, b, i16);
    CHECK_RESULT_TYPE_HOMO2(Float16, b, f16);
    CHECK_RESULT_TYPE_HOMO2(Float64, b, f64);
}

TEST(ResultTypeTest, Three) {
    // signed ints
    CHECK_RESULT_TYPE_HOMO3(Int32, i32, i32, i32);
    CHECK_RESULT_TYPE_HOMO3(Int32, i8, i8, i32);
    CHECK_RESULT_TYPE_HOMO3(Int32, i8, i16, i32);
    CHECK_RESULT_TYPE_HOMO3(Int32, i8, i32, i32);
    CHECK_RESULT_TYPE_HOMO3(Int32, i8, i32, i32);

    // unsigned ints
    CHECK_RESULT_TYPE_HOMO3(UInt8, u8, u8, u8);
    CHECK_RESULT_TYPE_HOMO3(Int16, u8, u8, i8);
    CHECK_RESULT_TYPE_HOMO3(Int16, u8, i8, i8);
    CHECK_RESULT_TYPE_HOMO3(Int16, u8, i8, i16);
    CHECK_RESULT_TYPE_HOMO3(Int16, u8, u8, i16);

    // float and signed int
    CHECK_RESULT_TYPE_HOMO3(Float16, f16, i8, i8);
    CHECK_RESULT_TYPE_HOMO3(Float16, f16, i32, i64);
    CHECK_RESULT_TYPE_HOMO3(Float32, f16, f32, i64);

    // float and unsigned int
    CHECK_RESULT_TYPE_HOMO3(Float16, f16, i8, u8);
    CHECK_RESULT_TYPE_HOMO3(Float16, f16, i16, u8);
    CHECK_RESULT_TYPE_HOMO3(Float32, f16, f32, u8);

    // bool and other
    CHECK_RESULT_TYPE_HOMO3(UInt8, b, u8, u8);
    CHECK_RESULT_TYPE_HOMO3(UInt8, b, b, u8);
    CHECK_RESULT_TYPE_HOMO3(Int16, b, i8, u8);
    CHECK_RESULT_TYPE_HOMO3(Int32, b, b, i32);
    CHECK_RESULT_TYPE_HOMO3(Float32, b, f16, f32);
    CHECK_RESULT_TYPE_HOMO3(Float64, b, b, f64);
}

TEST(ResultTypeTest, ArraysAndScalars) {
    // Arrays always take precedence

    // same dtype
    CHECK_RESULT_TYPE2(Int16, Ai16, Si16);
    // narrower vs wider
    CHECK_RESULT_TYPE2(Int16, Ai16, Si8);
    CHECK_RESULT_TYPE2(Int8, Ai8, Si16);
    // float vs int
    CHECK_RESULT_TYPE2(Float32, Af32, Si32);
    CHECK_RESULT_TYPE2(Int32, Ai32, Sf32);

    // 3 arguments
    CHECK_RESULT_TYPE3(Int8, Ai8, Si16, Si32);
    CHECK_RESULT_TYPE3(Int16, Ai16, Si8, Si32);
    CHECK_RESULT_TYPE3(Int16, Ai8, Ai16, Si32);
    CHECK_RESULT_TYPE3(Int8, Ai8, Si32, Sf64);
    CHECK_RESULT_TYPE3(Int8, Ai8, Sf32, Sf64);
    // unsigned
    CHECK_RESULT_TYPE3(UInt8, Au8, Si8, Si8);
    CHECK_RESULT_TYPE3(UInt8, Au8, Si8, Si16);
    CHECK_RESULT_TYPE3(Int16, Au8, Ai8, Si8);
    CHECK_RESULT_TYPE3(Int16, Au8, Ai8, Sf16);
    CHECK_RESULT_TYPE3(Float16, Au8, Af16, Si8);
    CHECK_RESULT_TYPE3(Int8, Ai8, Su8, Si8);
    // bool
    CHECK_RESULT_TYPE3(Bool, Ab, Sb, Sb);
    CHECK_RESULT_TYPE3(Bool, Ab, Si8, Si8);
    CHECK_RESULT_TYPE3(Bool, Ab, Sb, Si8);
    CHECK_RESULT_TYPE3(Int8, Ai8, Sb, Si8);
    CHECK_RESULT_TYPE3(Float32, Af32, Sb, Si8);
    CHECK_RESULT_TYPE3(Float32, Af32, Ab, Si8);
}

}  // namespace
}  // namespace chainerx
