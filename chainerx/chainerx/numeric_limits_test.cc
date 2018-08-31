#include "chainerx/numeric_limits.h"

#include <cstdint>
#include <limits>

namespace chainerx {

// LowestOrInf
static_assert(NumericLimits<bool>::LowestOrInf() == std::numeric_limits<bool>::lowest(), "");
static_assert(NumericLimits<uint8_t>::LowestOrInf() == std::numeric_limits<uint8_t>::lowest(), "");
static_assert(NumericLimits<int8_t>::LowestOrInf() == std::numeric_limits<int8_t>::lowest(), "");
static_assert(NumericLimits<int16_t>::LowestOrInf() == std::numeric_limits<int16_t>::lowest(), "");
static_assert(NumericLimits<int32_t>::LowestOrInf() == std::numeric_limits<int32_t>::lowest(), "");
static_assert(NumericLimits<int64_t>::LowestOrInf() == std::numeric_limits<int64_t>::lowest(), "");
static_assert(NumericLimits<float>::LowestOrInf() == -std::numeric_limits<float>::infinity(), "");
static_assert(NumericLimits<double>::LowestOrInf() == -std::numeric_limits<double>::infinity(), "");

// MaxOrInf
static_assert(NumericLimits<bool>::MaxOrInf() == std::numeric_limits<bool>::max(), "");
static_assert(NumericLimits<uint8_t>::MaxOrInf() == std::numeric_limits<uint8_t>::max(), "");
static_assert(NumericLimits<int8_t>::MaxOrInf() == std::numeric_limits<int8_t>::max(), "");
static_assert(NumericLimits<int16_t>::MaxOrInf() == std::numeric_limits<int16_t>::max(), "");
static_assert(NumericLimits<int32_t>::MaxOrInf() == std::numeric_limits<int32_t>::max(), "");
static_assert(NumericLimits<int64_t>::MaxOrInf() == std::numeric_limits<int64_t>::max(), "");
static_assert(NumericLimits<float>::MaxOrInf() == std::numeric_limits<float>::infinity(), "");
static_assert(NumericLimits<double>::MaxOrInf() == std::numeric_limits<double>::infinity(), "");

}  // namespace chainerx
