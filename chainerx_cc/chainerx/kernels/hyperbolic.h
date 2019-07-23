#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

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

}  // namespace chainerx
