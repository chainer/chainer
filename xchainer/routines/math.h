#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/scalar.h"

namespace xchainer {

namespace internal {

Array& IAdd(Array& x1, const Array& x2);
const Array& IAdd(const Array& x1, const Array& x2);

}  // namespace internal

Array Add(const Array& x1, const Array& x2);

namespace internal {

Array& IMultiply(Array& x1, const Array& x2);
const Array& IMultiply(const Array& x1, const Array& x2);

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2);
Array Multiply(const Array& x1, Scalar x2);
Array Multiply(Scalar x1, const Array& x2);

Array Sum(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false);

Array Maximum(const Array& x1, Scalar x2);
Array Maximum(Scalar x1, const Array& x2);

}  // namespace xchainer
