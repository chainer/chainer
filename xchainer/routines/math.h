#pragma once

#include <vector>

#include "xchainer/array.h"

namespace xchainer {
namespace routines {

Array& IAdd(Array& x1, const Array& x2);
const Array& IAdd(const Array& x1, const Array& x2);
Array Add(const Array& x1, const Array& x2);

Array& IMultiply(Array& x1, const Array& x2);
const Array& IMultiply(const Array& x1, const Array& x2);
Array Multiply(const Array& x1, const Array& x2);

Array Sum(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false);

}  // namespace routines
}  // namespace xchainer
