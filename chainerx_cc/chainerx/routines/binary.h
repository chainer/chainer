#pragma once

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace internal {

void IBitwiseAnd(const Array& x1, const Array& x2);
void IBitwiseAnd(const Array& x1, Scalar x2);

void IBitwiseOr(const Array& x1, const Array& x2);
void IBitwiseOr(const Array& x1, Scalar x2);

void IBitwiseXor(const Array& x1, const Array& x2);
void IBitwiseXor(const Array& x1, Scalar x2);

void ILeftShift(const Array& x1, const Array& x2);
void ILeftShift(const Array& x1, Scalar x2);

void IRightShift(const Array& x1, const Array& x2);
void IRightShift(const Array& x1, Scalar x2);

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

Array LeftShift(const Array& x1, const Array& x2);
Array LeftShift(const Array& x1, Scalar x2);
Array LeftShift(Scalar x1, const Array& x2);

Array RightShift(const Array& x1, const Array& x2);
Array RightShift(const Array& x1, Scalar x2);
Array RightShift(Scalar x1, const Array& x2);

}  // namespace chainerx
