#pragma once

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/kernel.h"

namespace chainerx {

class EqualKernel : public Kernel {
public:
    static const char* name() { return "Equal"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class NotEqualKernel : public Kernel {
public:
    static const char* name() { return "NotEqual"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class GreaterKernel : public Kernel {
public:
    static const char* name() { return "Greater"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class GreaterEqualKernel : public Kernel {
public:
    static const char* name() { return "GreaterEqual"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalNotKernel : public Kernel {
public:
    static const char* name() { return "LogicalNot"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class LogicalAndKernel : public Kernel {
public:
    static const char* name() { return "LogicalAnd"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalOrKernel : public Kernel {
public:
    static const char* name() { return "LogicalOr"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AllKernel : public Kernel {
public:
    static const char* name() { return "All"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class AnyKernel : public Kernel {
public:
    static const char* name() { return "Any"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

}  // namespace chainerx
