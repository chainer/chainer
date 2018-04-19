#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/ndim_vector.h"
#include "xchainer/scalar.h"

namespace xchainer {

Array Negative(const Array& x);

namespace internal {

Array& IAdd(Array& x1, const Array& x2);
const Array& IAdd(const Array& x1, const Array& x2);

}  // namespace internal

Array Add(const Array& x1, const Array& x2);

namespace internal {

Array& ISubtract(Array& x1, const Array& x2);
const Array& ISubtract(const Array& x1, const Array& x2);

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2);

namespace internal {

Array& IMultiply(Array& x1, const Array& x2);
const Array& IMultiply(const Array& x1, const Array& x2);

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2);
Array Multiply(const Array& x1, Scalar x2);
Array Multiply(Scalar x1, const Array& x2);

namespace internal {

Array& IDivide(Array& x1, const Array& x2);
const Array& IDivide(const Array& x1, const Array& x2);

}  // namespace internal

Array Divide(const Array& x1, const Array& x2);

Array Sum(const Array& a, const nonstd::optional<NdimVector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false);
// TODO(niboshi): Move to statistics routines
Array AMax(const Array& a, const nonstd::optional<NdimVector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false);

Array Maximum(const Array& x1, Scalar x2);
Array Maximum(Scalar x1, const Array& x2);

Array Exp(const Array& x);
Array Log(const Array& x);

// Returns the LogSumExp (LSE) of x, reduced along the specified axes.
// If no axes are specified, all axes will be reduced.
Array LogSumExp(const Array& x, const nonstd::optional<NdimVector<int8_t>>& axis = nonstd::nullopt, bool keepdims = false);

// Returns the logarithm of the softmax of x along the specified axes.
// If no axes are specified, the softmax is applied on the second axis.
Array LogSoftmax(const Array& x, const nonstd::optional<NdimVector<int8_t>>& axis = nonstd::nullopt);

}  // namespace xchainer
