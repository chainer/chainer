#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

class SinKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class CosKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class TanKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArcsinKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArccosKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArctanKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class Arctan2Kernel : public Kernel {
public:
    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

}  // namespace chainerx
