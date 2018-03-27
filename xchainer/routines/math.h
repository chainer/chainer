#pragma once

#include "xchainer/array.h"

namespace xchainer {
namespace routines {

Array& IAdd(Array& lhs, const Array& rhs);
const Array& IAdd(const Array& lhs, const Array& rhs);
Array Add(const Array& lhs, const Array& rhs);

Array& IMul(Array& lhs, const Array& rhs);
const Array& IMul(const Array& lhs, const Array& rhs);
Array Mul(const Array& lhs, const Array& rhs);

Array Sum(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false);

}  // namespace routines
}  // namespace xchainer
