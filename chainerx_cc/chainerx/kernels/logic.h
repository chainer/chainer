#pragma once

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/op.h"

namespace chainerx {

class EqualOp : public Op {
public:
    static const char* name() { return "Equal"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class NotEqualOp : public Op {
public:
    static const char* name() { return "NotEqual"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class GreaterOp : public Op {
public:
    static const char* name() { return "Greater"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class GreaterEqualOp : public Op {
public:
    static const char* name() { return "GreaterEqual"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalNotOp : public Op {
public:
    static const char* name() { return "LogicalNot"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class LogicalAndOp : public Op {
public:
    static const char* name() { return "LogicalAnd"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class LogicalOrOp : public Op {
public:
    static const char* name() { return "LogicalOr"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AllOp : public Op {
public:
    static const char* name() { return "All"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class AnyOp : public Op {
public:
    static const char* name() { return "Any"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

}  // namespace chainerx
