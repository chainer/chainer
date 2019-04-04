#pragma once

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/device.h"
#include "chainerx/op.h"

namespace chainerx {

// Returns an elementwise equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
class EqualOp : public Op {
public:
    static const char* name() { return "Equal"; }

    virtual Array Call(const Array& x1, const Array& x2);

protected:
    virtual void Impl(const Array& x1, const Array& x2, const Array& out) = 0;
};

inline Array Equal(const Array& x1, const Array& x2) { return x1.device().backend().CallOp<EqualOp>(x1, x2); }

// Returns an elementwise non-equality array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
class NotEqualOp : public Op {
public:
    static const char* name() { return "NotEqual"; }

    virtual Array Call(const Array& x1, const Array& x2);

protected:
    virtual void Impl(const Array& x1, const Array& x2, const Array& out) = 0;
};

inline Array NotEqual(const Array& x1, const Array& x2) { return x1.device().backend().CallOp<NotEqualOp>(x1, x2); }

// Returns an elementwise `x1` > `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
class GreaterOp : public Op {
public:
    static const char* name() { return "Greater"; }

    virtual Array Call(const Array& x1, const Array& x2);

protected:
    virtual void Impl(const Array& x1, const Array& x2, const Array& out) = 0;
};

inline Array Greater(const Array& x1, const Array& x2) { return x1.device().backend().CallOp<GreaterOp>(x1, x2); }

// Returns an elementwise `x1` >= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
class GreaterEqualOp : public Op {
public:
    static const char* name() { return "GreaterEqual"; }

    virtual Array Call(const Array& x1, const Array& x2);

protected:
    virtual void Impl(const Array& x1, const Array& x2, const Array& out) = 0;
};

inline Array GreaterEqual(const Array& x1, const Array& x2) { return x1.device().backend().CallOp<GreaterEqualOp>(x1, x2); }

// Returns an elementwise `x1` < `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
inline Array Less(const Array& x1, const Array& x2) { return Greater(x2, x1); }

// Returns an elementwise `x1` <= `x2` array.
//
// Dtype casting is not supported: if x1 and x2 have different types, DtypeError is thrown.
inline Array LessEqual(const Array& x1, const Array& x2) { return GreaterEqual(x2, x1); }

// Returns an elementwise logical negation of an array.
class LogicalNotOp : public Op {
public:
    static const char* name() { return "LogicalNot"; }

    virtual Array Call(const Array& x);

protected:
    virtual void Impl(const Array& x, const Array& out) = 0;
};

inline Array LogicalNot(const Array& x) { return x.device().backend().CallOp<LogicalNotOp>(x); }

}  // namespace chainerx
