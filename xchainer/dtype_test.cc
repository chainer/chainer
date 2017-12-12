#include "xchainer/dtype.h"

#include <gtest/gtest.h>
#include <type_traits>

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

}  // namespace
}  // namespace xchainer
