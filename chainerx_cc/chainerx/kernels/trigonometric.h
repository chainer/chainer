#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

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

}  // namespace chainerx
