#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class FillKernel : public Kernel {
public:
    static const char* name() { return "Fill"; }

    virtual void Call(const Array& out, Scalar value) = 0;
};

// Casts the elements from one array to the other dtype, and store into the other.
class AsTypeKernel : public Kernel {
public:
    static const char* name() { return "AsType"; }

    virtual void Call(const Array& a, const Array& out) = 0;
};

class SqrtKernel : public Kernel {
public:
    static const char* name() { return "Sqrt"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SquareKernel : public Kernel {
public:
    static const char* name() { return "Square"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class FabsKernel : public Kernel {
public:
    static const char* name() { return "Fabs"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SignKernel : public Kernel {
public:
    static const char* name() { return "Sign"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

// Compares x1 and x2 and assign either pos or neg according to the result.
// Formally, it calculates: out = x1 < x2 ? pos : neg
class IfLessElseASSAKernel : public Kernel {
public:
    static const char* name() { return "IfLessElseASSA"; }

    virtual void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) = 0;
};

// Compares x1 and x2 and assign either pos or neg according to the result.
// Formally, it calculates: out = x1 > x2 ? pos : neg
class IfGreaterElseASSAKernel : public Kernel {
public:
    static const char* name() { return "IfGreaterElseASSA"; }

    virtual void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) = 0;
};

class IfGreaterElseAAAAKernel : public Kernel {
public:
    static const char* name() { return "IfGreaterElseAAAA"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) = 0;
};

}  // namespace chainerx
