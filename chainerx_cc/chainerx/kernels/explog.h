#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

class ExpKernel : public Kernel {
public:
    static const char* name() { return "Exp"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class Expm1Kernel : public Kernel {
public:
    static const char* name() { return "Expm1"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class Exp2Kernel : public Kernel {
public:
    static const char* name() { return "Exp2"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class LogKernel : public Kernel {
public:
    static const char* name() { return "Log"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class Log10Kernel : public Kernel {
public:
    static const char* name() { return "Log10"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class Log1pKernel : public Kernel {
public:
    static const char* name() { return "Log1p"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

}  // namespace chainerx
