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
    ASSERT_EQ(dtype, VisitDtype(dtype, [](auto pt) { return decltype(pt)::kDtype; }));
}

// Check if GetDtypeName() and GetDtype(std::string) are inverses of each other.
TEST_P(AllDtypeTest, GetDtypeAndGetDtypeName) {
    const Dtype dtype = GetParam();
    ASSERT_EQ(dtype, GetDtype(GetDtypeName(dtype)));
}

// Check if char GetCharCode() and GetDtype(std::string) are inverses of each other.
TEST_P(AllDtypeTest, GetDtypeAndGetCharCode) {
    const Dtype dtype = GetParam();
    ASSERT_EQ(dtype, GetDtype({GetCharCode(dtype)}));
}

INSTANTIATE_TEST_CASE_P(TestWithAllDtypes, AllDtypeTest, ::testing::ValuesIn(GetAllDtypes()));

TEST(DtypeTest, WrongDtypeName) { ASSERT_THROW(GetDtype("wrong"), DtypeError); }

TEST(DtypeTest, CheckEqual) {
    ASSERT_NO_THROW(CheckEqual(Dtype::kInt8, Dtype::kInt8));
    ASSERT_THROW(CheckEqual(Dtype::kInt8, Dtype::kUInt8), DtypeError);
}

}  // namespace
}  // namespace xchainer
