#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

Array Negative(const Array& x);

namespace internal {

void IAdd(const Array& x1, const Array& x2);
void IAdd(const Array& x1, Scalar x2);

}  // namespace internal

Array Add(const Array& x1, const Array& x2);
Array Add(const Array& x1, Scalar x2);
Array Add(Scalar x1, const Array& x2);

namespace internal {

void ISubtract(const Array& x1, const Array& x2);
void ISubtract(const Array& x1, Scalar x2);

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2);
Array Subtract(const Array& x1, Scalar x2);
Array Subtract(Scalar x1, const Array& x2);

namespace internal {

void IMultiply(const Array& x1, const Array& x2);
void IMultiply(const Array& x1, Scalar x2);

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2);
Array Multiply(const Array& x1, Scalar x2);
Array Multiply(Scalar x1, const Array& x2);

namespace internal {

void IFloorDivide(const Array& x1, const Array& x2);
void IFloorDivide(const Array& x1, Scalar x2);

void ITrueDivide(const Array& x1, const Array& x2);
void ITrueDivide(const Array& x1, Scalar x2);

void IDivide(const Array& x1, const Array& x2);
void IDivide(const Array& x1, Scalar x2);

}  // namespace internal

Array FloorDivide(const Array& x1, const Array& x2);
Array FloorDivide(const Array& x1, Scalar x2);
Array FloorDivide(Scalar x1, const Array& x2);

Array Divide(const Array& x1, const Array& x2);
Array Divide(const Array& x1, Scalar x2);
Array Divide(Scalar x1, const Array& x2);

Array TrueDivide(const Array& x1, const Array& x2);
Array TrueDivide(const Array& x1, Scalar x2);
Array TrueDivide(Scalar x1, const Array& x2);

Array Reciprocal(const Array& x);

Array Maximum(const Array& x1, Scalar x2);
Array Maximum(Scalar x1, const Array& x2);
Array Maximum(const Array& x1, const Array& x2);

Array Minimum(const Array& x1, Scalar x2);
Array Minimum(Scalar x1, const Array& x2);
Array Minimum(const Array& x1, const Array& x2);

Array Exp(const Array& x);
Array Expm1(const Array& x);
Array Exp2(const Array& x);
Array Log(const Array& x);
Array Log10(const Array& x);
Array Log1p(const Array& x);

Array Sigmoid(const Array& x);

Array Relu(const Array& x);

Array Square(const Array& x);

Array SquaredDifference(const Array& x1, const Array& x2);

Array Sqrt(const Array& x);

Array Power(const Array& x1, const Array& x2);

Array Power(const Array& x1, Scalar x2);

Array Power(Scalar x1, const Array& x2);

Array IsNan(const Array& x);

Array IsInf(const Array& x);

Array IsFinite(const Array& x);

Array Absolute(const Array& x);

Array Ceil(const Array& x);

Array Floor(const Array& x);

Array Fabs(const Array& x);

Array Sign(const Array& x);

}  // namespace chainerx
