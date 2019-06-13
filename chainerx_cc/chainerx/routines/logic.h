#pragma once

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/device.h"
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

// Returns an elementwise equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Equal(const Array& x1, const Array& x2);

// Returns an elementwise non-equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array NotEqual(const Array& x1, const Array& x2);

// Returns an elementwise `x1` > `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array Greater(const Array& x1, const Array& x2);

// Returns an elementwise `x1` >= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
Array GreaterEqual(const Array& x1, const Array& x2);

// Returns an elementwise `x1` < `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
inline Array Less(const Array& x1, const Array& x2) { return Greater(x2, x1); }

// Returns an elementwise `x1` <= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
inline Array LessEqual(const Array& x1, const Array& x2) { return GreaterEqual(x2, x1); }

// Returns an elementwise logical negation of an array.
Array LogicalNot(const Array& x);

Array LogicalAnd(const Array& x1, const Array& x2);

Array LogicalOr(const Array& x1, const Array& x2);

}  // namespace chainerx
