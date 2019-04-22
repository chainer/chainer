#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class AddKernel : public Kernel {
public:
    static const char* name() { return "Add"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AddASKernel : public Kernel {
public:
    static const char* name() { return "AddAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class SubtractKernel : public Kernel {
public:
    static const char* name() { return "Subtract"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class SubtractASKernel : public Kernel {
public:
    static const char* name() { return "SubtractAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class MultiplyKernel : public Kernel {
public:
    static const char* name() { return "Multiply"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class MultiplyASKernel : public Kernel {
public:
    static const char* name() { return "MultiplyAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class FloorDivideKernel : public Kernel {
public:
    static const char* name() { return "FloorDivide"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class FloorDivideASKernel : public Kernel {
public:
    static const char* name() { return "FloorDivideAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class DivideKernel : public Kernel {
public:
    static const char* name() { return "Divide"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class DivideASKernel : public Kernel {
public:
    static const char* name() { return "DivideAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class ExpKernel : public Kernel {
public:
    static const char* name() { return "Exp"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class LogKernel : public Kernel {
public:
    static const char* name() { return "Log"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SquareKernel : public Kernel {
public:
    static const char* name() { return "Square"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SqrtKernel : public Kernel {
public:
    static const char* name() { return "Sqrt"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SinKernel : public Kernel {
public:
    static const char* name() { return "Sin"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class CosKernel : public Kernel {
public:
    static const char* name() { return "Cos"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class TanKernel : public Kernel {
public:
    static const char* name() { return "Tan"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SinhKernel : public Kernel {
public:
    static const char* name() { return "Sinh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class CoshKernel : public Kernel {
public:
    static const char* name() { return "Cosh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class TanhKernel : public Kernel {
public:
    static const char* name() { return "Tanh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArcsinKernel : public Kernel {
public:
    static const char* name() { return "Arcsin"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArccosKernel : public Kernel {
public:
    static const char* name() { return "Arccos"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArctanKernel : public Kernel {
public:
    static const char* name() { return "Arctan"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class Arctan2Kernel : public Kernel {
public:
    static const char* name() { return "Arctan2"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class ArcsinhKernel : public Kernel {
public:
    static const char* name() { return "Archsinh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArccoshKernel : public Kernel {
public:
    static const char* name() { return "Arccosh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class CeilKernel : public Kernel {
public:
    static const char* name() { return "Ceil"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class FloorKernel : public Kernel {
public:
    static const char* name() { return "Floor"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class IsNanKernel : public Kernel {
public:
    static const char* name() { return "IsNan"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class IsInfKernel : public Kernel {
public:
    static const char* name() { return "IsInf"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

// Calculate the sum of an array.
// It will be summed over the specified axes.
// `axis` must be normalized so that
// - it has only positive values,
// - it is sorted, and
// - it has no duplicated values.
// Otherwise, the behavior is undefined.
class SumKernel : public Kernel {
public:
    static const char* name() { return "Sum"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

// Calculates the maximum along specified axes.
// See Sum() for the explanation of arguments.
class AMaxKernel : public Kernel {
public:
    static const char* name() { return "AMax"; }

    virtual void Call(const Array& src, const Axes& axis, const Array& out) = 0;
};

// Calculates the minimum along specified axes.
// See Sum() for the explanation of arguments.
class AMinKernel : public Kernel {
public:
    static const char* name() { return "AMin"; }

    virtual void Call(const Array& src, const Axes& axis, const Array& out) = 0;
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
