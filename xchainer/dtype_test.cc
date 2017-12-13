#include "xchainer/dtype.h"

#include <type_traits>

#include <gtest/gtest.h>

namespace xchainer {
namespace {

// Check if DtypeToType and TypeToDtype are inverses of each other.
template <Dtype dtype>
constexpr bool kDtypeMappingTest = dtype == TypeToDtype<DtypeToType<dtype>>;

static_assert(kDtypeMappingTest<Dtype::kBool>, "bool");
static_assert(kDtypeMappingTest<Dtype::kInt8>, "int8");
static_assert(kDtypeMappingTest<Dtype::kInt16>, "int16");
static_assert(kDtypeMappingTest<Dtype::kInt32>, "int32");
static_assert(kDtypeMappingTest<Dtype::kInt64>, "int64");
static_assert(kDtypeMappingTest<Dtype::kUInt8>, "uint8");
static_assert(kDtypeMappingTest<Dtype::kFloat32>, "float32");
static_assert(kDtypeMappingTest<Dtype::kFloat64>, "float64");

// Check if GetCharCode and CharToDtype are inverses of each other.
template <Dtype dtype>
constexpr bool kDtypeCharMappingTest = dtype == CharToDtype<GetCharCode(dtype)>;

static_assert(kDtypeCharMappingTest<Dtype::kBool>, "bool");
static_assert(kDtypeCharMappingTest<Dtype::kInt8>, "int8");
static_assert(kDtypeCharMappingTest<Dtype::kInt16>, "int16");
static_assert(kDtypeCharMappingTest<Dtype::kInt32>, "int32");
static_assert(kDtypeCharMappingTest<Dtype::kInt64>, "int64");
static_assert(kDtypeCharMappingTest<Dtype::kUInt8>, "uint8");
static_assert(kDtypeCharMappingTest<Dtype::kFloat32>, "float32");
static_assert(kDtypeCharMappingTest<Dtype::kFloat64>, "float64");

// Check if the element size is correct.
template <typename T>
constexpr bool kGetElementSizeTest = GetElementSize(TypeToDtype<T>) == sizeof(T);

static_assert(kGetElementSizeTest<bool>, "bool");
static_assert(kGetElementSizeTest<int8_t>, "int8");
static_assert(kGetElementSizeTest<int16_t>, "int16");
static_assert(kGetElementSizeTest<int32_t>, "int32");
static_assert(kGetElementSizeTest<int64_t>, "int64");
static_assert(kGetElementSizeTest<uint8_t>, "uint8");
static_assert(kGetElementSizeTest<float>, "float32");
static_assert(kGetElementSizeTest<double>, "float64");

// Check if char* GetDtypeName and GetDtype(std::string) are inverses of each other.
TEST(DtypeTest, GetDtype_GetDtypeName) {
    ASSERT_EQ(Dtype::kBool, GetDtype(GetDtypeName(Dtype::kBool)));
    ASSERT_EQ(Dtype::kInt8, GetDtype(GetDtypeName(Dtype::kInt8)));
    ASSERT_EQ(Dtype::kInt16, GetDtype(GetDtypeName(Dtype::kInt16)));
    ASSERT_EQ(Dtype::kInt32, GetDtype(GetDtypeName(Dtype::kInt32)));
    ASSERT_EQ(Dtype::kInt64, GetDtype(GetDtypeName(Dtype::kInt64)));
    ASSERT_EQ(Dtype::kUInt8, GetDtype(GetDtypeName(Dtype::kUInt8)));
    ASSERT_EQ(Dtype::kFloat32, GetDtype(GetDtypeName(Dtype::kFloat32)));
    ASSERT_EQ(Dtype::kFloat64, GetDtype(GetDtypeName(Dtype::kFloat64)));
    ASSERT_THROW(GetDtype("wrong"), DtypeError);
}

// Check if char GetCharCode and GetDtype(std::string) are inverses of each other.
TEST(DtypeTest, GetDtype_GetCharCode) {
    ASSERT_EQ(Dtype::kBool, GetDtype({GetCharCode(Dtype::kBool)}));
    ASSERT_EQ(Dtype::kInt8, GetDtype({GetCharCode(Dtype::kInt8)}));
    ASSERT_EQ(Dtype::kInt16, GetDtype({GetCharCode(Dtype::kInt16)}));
    ASSERT_EQ(Dtype::kInt32, GetDtype({GetCharCode(Dtype::kInt32)}));
    ASSERT_EQ(Dtype::kInt64, GetDtype({GetCharCode(Dtype::kInt64)}));
    ASSERT_EQ(Dtype::kUInt8, GetDtype({GetCharCode(Dtype::kUInt8)}));
    ASSERT_EQ(Dtype::kFloat32, GetDtype({GetCharCode(Dtype::kFloat32)}));
    ASSERT_EQ(Dtype::kFloat64, GetDtype({GetCharCode(Dtype::kFloat64)}));
}

TEST(DtypeTest, CheckEqual) {
    ASSERT_NO_THROW(CheckEqual(Dtype::kInt8, Dtype::kInt8));
    ASSERT_THROW(CheckEqual(Dtype::kInt8, Dtype::kUInt8), DtypeError);
}

}  // namespace
}  // namespace xchainer
