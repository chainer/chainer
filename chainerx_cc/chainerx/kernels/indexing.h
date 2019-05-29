#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/kernel.h"

namespace chainerx {

class AddAtKernel : public Kernel {
public:
    static const char* name() { return "AddAt"; }

    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) = 0;
};

class TakeKernel : public Kernel {
public:
    static const char* name() { return "Take"; }

    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& out) = 0;
};

class WhereKernel : public Kernel {
public:
    static const char* name() { return "Where"; }

    virtual void Call(const Array& condition, const Array& x, const Array& y, const Array& out) = 0;
};

class WhereASKernel : public Kernel {
public:
    static const char* name() { return "WhereAS"; }

    virtual void Call(const Array& condition, const Array& x, Scalar y, const Array& out) = 0;
};

class WhereSAKernel : public Kernel {
public:
    static const char* name() { return "WhereSA"; }

    virtual void Call(const Array& condition, Scalar x, const Array& y, const Array& out) = 0;
};

class WhereSSKernel : public Kernel {
public:
    static const char* name() { return "WhereSS"; }

    virtual void Call(const Array& condition, Scalar x, Scalar y, const Array& out) = 0;
};

}  // namespace chainerx
