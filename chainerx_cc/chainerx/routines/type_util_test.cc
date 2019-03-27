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

TEST(DtypeTest, GetDefaultDtype) {
    EXPECT_EQ(Dtype::kBool, internal::GetDefaultDtype(DtypeKind::kBool));
    EXPECT_EQ(Dtype::kInt32, internal::GetDefaultDtype(DtypeKind::kInt));
    EXPECT_EQ(Dtype::kFloat32, internal::GetDefaultDtype(DtypeKind::kFloat));
}

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
        Scalar Sb{bool{1}};                          \
        Scalar Si{int64_t{1}};                       \
        Scalar Sf{double{1}};                        \
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

TEST(ResultTypeTest, NoArgs) {
    EXPECT_THROW({ chainerx::ResultType(std::vector<Array>{}); }, ChainerxError);
}

TEST(ResultTypeTest, One) {
    CHECK_RESULT_TYPE1(Ab);
    CHECK_RESULT_TYPE1(Af16);
    CHECK_RESULT_TYPE1(Af32);
    CHECK_RESULT_TYPE1(Af64);
    CHECK_RESULT_TYPE1(Ai8);
    CHECK_RESULT_TYPE1(Ai16);
    CHECK_RESULT_TYPE1(Ai32);
    CHECK_RESULT_TYPE1(Ai64);
    CHECK_RESULT_TYPE1(Au8);
}

TEST(ResultTypeTest, TwoBool) { CHECK_RESULT_TYPE2(Bool, Ab, Ab); }

TEST(ResultTypeTest, TwoFloat) {
    CHECK_RESULT_TYPE2(Float16, Af16, Af16);
    CHECK_RESULT_TYPE2(Float32, Af32, Af32);
    CHECK_RESULT_TYPE2(Float64, Af64, Af64);
    CHECK_RESULT_TYPE2(Float32, Af16, Af32);
    CHECK_RESULT_TYPE2(Float64, Af16, Af64);
    CHECK_RESULT_TYPE2(Float64, Af32, Af64);
}

TEST(ResultTypeTest, TwoSignedInt) {
    CHECK_RESULT_TYPE2(Int8, Ai8, Ai8);
    CHECK_RESULT_TYPE2(Int16, Ai8, Ai16);
    CHECK_RESULT_TYPE2(Int32, Ai8, Ai32);
    CHECK_RESULT_TYPE2(Int64, Ai8, Ai64);
    CHECK_RESULT_TYPE2(Int16, Ai16, Ai16);
    CHECK_RESULT_TYPE2(Int32, Ai32, Ai32);
    CHECK_RESULT_TYPE2(Int64, Ai64, Ai64);
    CHECK_RESULT_TYPE2(Int32, Ai16, Ai32);
    CHECK_RESULT_TYPE2(Int64, Ai16, Ai64);
    CHECK_RESULT_TYPE2(Int64, Ai32, Ai64);
}

TEST(ResultTypeTest, TwoUnsignedInt) { CHECK_RESULT_TYPE2(UInt8, Au8, Au8); }

TEST(ResultTypeTest, TwoSignedIntAndUnsignedInt) {
    CHECK_RESULT_TYPE2(Int16, Au8, Ai8);
    CHECK_RESULT_TYPE2(Int16, Au8, Ai16);
    CHECK_RESULT_TYPE2(Int32, Au8, Ai32);
}

TEST(ResultTypeTest, TwoIntAndFloat) {
    CHECK_RESULT_TYPE2(Float16, Ai8, Af16);
    CHECK_RESULT_TYPE2(Float16, Au8, Af16);
    CHECK_RESULT_TYPE2(Float32, Ai16, Af32);
    CHECK_RESULT_TYPE2(Float32, Ai32, Af32);
    CHECK_RESULT_TYPE2(Float32, Ai64, Af32);
}

TEST(ResultTypeTest, TwoBoolAndOther) {
    CHECK_RESULT_TYPE2(UInt8, Ab, Au8);
    CHECK_RESULT_TYPE2(Int8, Ab, Ai8);
    CHECK_RESULT_TYPE2(Int16, Ab, Ai16);
    CHECK_RESULT_TYPE2(Float16, Ab, Af16);
    CHECK_RESULT_TYPE2(Float64, Ab, Af64);
}

TEST(ResultTypeTest, Three) {
    // signed ints
    CHECK_RESULT_TYPE3(Int32, Ai32, Ai32, Ai32);
    CHECK_RESULT_TYPE3(Int32, Ai8, Ai8, Ai32);
    CHECK_RESULT_TYPE3(Int32, Ai8, Ai16, Ai32);
    CHECK_RESULT_TYPE3(Int32, Ai8, Ai32, Ai32);

    // unsigned ints
    CHECK_RESULT_TYPE3(UInt8, Au8, Au8, Au8);
    CHECK_RESULT_TYPE3(Int16, Au8, Au8, Ai8);
    CHECK_RESULT_TYPE3(Int16, Au8, Ai8, Ai8);
    CHECK_RESULT_TYPE3(Int16, Au8, Ai8, Ai16);
    CHECK_RESULT_TYPE3(Int16, Au8, Au8, Ai16);

    // float and signed int
    CHECK_RESULT_TYPE3(Float16, Af16, Ai8, Ai8);
    CHECK_RESULT_TYPE3(Float16, Af16, Ai32, Ai64);
    CHECK_RESULT_TYPE3(Float32, Af16, Af32, Ai64);

    // float and unsigned int
    CHECK_RESULT_TYPE3(Float16, Af16, Ai8, Au8);
    CHECK_RESULT_TYPE3(Float16, Af16, Ai16, Au8);
    CHECK_RESULT_TYPE3(Float32, Af16, Af32, Au8);

    // bool and other
    CHECK_RESULT_TYPE3(UInt8, Ab, Au8, Au8);
    CHECK_RESULT_TYPE3(UInt8, Ab, Ab, Au8);
    CHECK_RESULT_TYPE3(Int16, Ab, Ai8, Au8);
    CHECK_RESULT_TYPE3(Int32, Ab, Ab, Ai32);
    CHECK_RESULT_TYPE3(Float32, Ab, Af16, Af32);
    CHECK_RESULT_TYPE3(Float64, Ab, Ab, Af64);
}

TEST(ResultTypeTest, ArraysAndScalars) {
    // Arrays take precedence unless Scalar is a wider floating kind.

    // ints
    CHECK_RESULT_TYPE2(Int8, Ai8, Si);
    CHECK_RESULT_TYPE2(Int16, Ai16, Si);
    // float vs int
    CHECK_RESULT_TYPE2(Float32, Af32, Si);
    CHECK_RESULT_TYPE2(Float32, Ai32, Sf);

    // 3 arguments
    CHECK_RESULT_TYPE3(Int8, Ai8, Si, Si);
    CHECK_RESULT_TYPE3(Int16, Ai8, Ai16, Si);
    CHECK_RESULT_TYPE3(Float32, Ai8, Si, Sf);
    // unsigned
    CHECK_RESULT_TYPE3(UInt8, Au8, Si, Si);
    CHECK_RESULT_TYPE3(Int8, Ai8, Si, Si);
    CHECK_RESULT_TYPE3(Int16, Au8, Ai8, Si);
    CHECK_RESULT_TYPE3(Float32, Au8, Ai8, Sf);
    CHECK_RESULT_TYPE3(Float16, Au8, Af16, Si);
    // bool
    CHECK_RESULT_TYPE3(Bool, Ab, Sb, Sb);
    CHECK_RESULT_TYPE3(Bool, Ab, Si, Si);
    CHECK_RESULT_TYPE3(Bool, Ab, Sb, Si);
    CHECK_RESULT_TYPE3(Int8, Ai8, Sb, Si);
    CHECK_RESULT_TYPE3(Float32, Af32, Sb, Si);
    CHECK_RESULT_TYPE3(Float32, Af32, Ab, Si);
    CHECK_RESULT_TYPE3(Float32, Ab, Ab, Sf);
    CHECK_RESULT_TYPE3(Float16, Ab, Af16, Sf);
}

}  // namespace
}  // namespace chainerx
