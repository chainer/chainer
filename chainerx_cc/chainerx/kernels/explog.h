#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

class ErfKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class ExpKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class Expm1Kernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class Exp2Kernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class LogKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class Log10Kernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class Log2Kernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class Log1pKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

}  // namespace chainerx
