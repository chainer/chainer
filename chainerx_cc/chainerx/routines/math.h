#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array Reciprocal(const Array& x);

Array Sum(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);
// TODO(niboshi): Move to statistics routines
Array AMax(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);
Array AMin(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

Array Maximum(const Array& x1, Scalar x2);
Array Maximum(Scalar x1, const Array& x2);
Array Maximum(const Array& x1, const Array& x2);

Array Minimum(const Array& x1, Scalar x2);
Array Minimum(Scalar x1, const Array& x2);
Array Minimum(const Array& x1, const Array& x2);

Array Exp(const Array& x);
Array Log(const Array& x);
Array Log10(const Array& x);

// Returns the LogSumExp (LSE) of x, reduced along the specified axes.
// If no axes are specified, all axes will be reduced.
Array LogSumExp(const Array& x, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

// Returns the logarithm of the softmax of x along the specified axes.
// If no axes are specified, the softmax is applied on the second axis.
Array LogSoftmax(const Array& x, const OptionalAxes& axis = nonstd::nullopt);

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

Array Softmax(const Array& x, const OptionalAxes& axis = nonstd::nullopt);

Array Square(const Array& x);

Array SquaredDifference(const Array& x1, const Array& x2);

Array Sqrt(const Array& x);

Array IsNan(const Array& x);

Array IsInf(const Array& x);

Array IsFinite(const Array& x);

Array Tanh(const Array& x);

Array Sin(const Array& x);

Array Cos(const Array& x);

Array Absolute(const Array& x);

Array Tan(const Array& x);

Array Arcsin(const Array& x);

Array Arccos(const Array& x);

Array Arctan(const Array& x);

Array Arctan2(const Array& x1, const Array& x2);

Array Sinh(const Array& x);

Array Cosh(const Array& x);

Array Arcsinh(const Array& x);

Array Arccosh(const Array& x);

Array Ceil(const Array& x);

Array Floor(const Array& x);

namespace internal {

void IBitwiseAnd(const Array& x1, const Array& x2);
void IBitwiseAnd(const Array& x1, Scalar x2);

void IBitwiseOr(const Array& x1, const Array& x2);
void IBitwiseOr(const Array& x1, Scalar x2);

void IBitwiseXor(const Array& x1, const Array& x2);
void IBitwiseXor(const Array& x1, Scalar x2);

}  // namespace internal

Array BitwiseAnd(const Array& x1, const Array& x2);
Array BitwiseAnd(const Array& x1, Scalar x2);
Array BitwiseAnd(Scalar x1, const Array& x2);

Array BitwiseOr(const Array& x1, const Array& x2);
Array BitwiseOr(const Array& x1, Scalar x2);
Array BitwiseOr(Scalar x1, const Array& x2);

Array BitwiseXor(const Array& x1, const Array& x2);
Array BitwiseXor(const Array& x1, Scalar x2);
Array BitwiseXor(Scalar x1, const Array& x2);

}  // namespace chainerx
