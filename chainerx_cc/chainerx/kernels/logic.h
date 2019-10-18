#pragma once

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/kernel.h"

namespace chainerx {

class EqualKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class NotEqualKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class GreaterKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class GreaterEqualKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalNotKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class LogicalAndKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalOrKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalXorKernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AllKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class AnyKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class IsNanKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class IsInfKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class IsFiniteKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

}  // namespace chainerx
