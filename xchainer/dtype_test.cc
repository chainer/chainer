#include "xchainer/dtype.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace xchainer {
namespace {

// Tests on all dtypes
class AllDtypeTest : public ::testing::TestWithParam<Dtype> {};

// Check if mapping from dtype to primitive type (provided by VisitDtype) and the opposite mapping by PrimitiveType<T> are consistent.
TEST_P(AllDtypeTest, DtypeMapping) {
    const Dtype dtype = GetParam();
    EXPECT_EQ(dtype, VisitDtype(dtype, [](auto pt) { return decltype(pt)::kDtype; }));
}

// Check if GetDtypeName() and GetDtype(std::string) are inverses of each other.
TEST_P(AllDtypeTest, GetDtypeAndGetDtypeName) {
    const Dtype dtype = GetParam();
    EXPECT_EQ(dtype, GetDtype(GetDtypeName(dtype)));
}

// Check if char GetCharCode() and GetDtype(std::string) are inverses of each other.
TEST_P(AllDtypeTest, GetDtypeAndGetCharCode) {
    const Dtype dtype = GetParam();
    EXPECT_EQ(dtype, GetDtype({GetCharCode(dtype)}));
}

INSTANTIATE_TEST_CASE_P(TestWithAllDtypes, AllDtypeTest, ::testing::ValuesIn(GetAllDtypes()));

TEST(DtypeTest, WrongDtypeName) { EXPECT_THROW(GetDtype("wrong"), DtypeError); }

TEST(DtypeTest, CheckEqual) {
    EXPECT_NO_THROW(CheckEqual(Dtype::kInt8, Dtype::kInt8));
    EXPECT_THROW(CheckEqual(Dtype::kInt8, Dtype::kUInt8), DtypeError);
}

TEST(DtypeTest, IsValidDtype) {
    EXPECT_TRUE(IsValidDtype(Dtype::kBool));
    EXPECT_TRUE(IsValidDtype(Dtype::kInt8));
    EXPECT_TRUE(IsValidDtype(Dtype::kInt16));
    EXPECT_TRUE(IsValidDtype(Dtype::kInt32));
    EXPECT_TRUE(IsValidDtype(Dtype::kInt64));
    EXPECT_TRUE(IsValidDtype(Dtype::kUInt8));
    EXPECT_TRUE(IsValidDtype(Dtype::kFloat32));
    EXPECT_TRUE(IsValidDtype(Dtype::kFloat64));
    EXPECT_FALSE(IsValidDtype(static_cast<Dtype>(0)));
    EXPECT_FALSE(IsValidDtype(static_cast<Dtype>(-1)));
    EXPECT_FALSE(IsValidDtype(static_cast<Dtype>(static_cast<int>(Dtype::kFloat64) + 1)));
}

}  // namespace
}  // namespace xchainer
