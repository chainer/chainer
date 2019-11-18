#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class FillKernel : public Kernel {
public:
    virtual void Call(const Array& out, Scalar value) = 0;
};

class SqrtKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class SquareKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class AbsKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class SignKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

// Compares x1 and x2 and assign either pos or neg according to the result.
// Formally, it calculates: out = x1 < x2 ? pos : neg
class IfLessElseASSAKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) = 0;
};

// Compares x1 and x2 and assign either pos or neg according to the result.
// Formally, it calculates: out = x1 > x2 ? pos : neg
class IfGreaterElseASSAKernel : public Kernel {
public:
    virtual void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) = 0;
};

class IfGreaterElseAAAAKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) = 0;
};

}  // namespace chainerx
