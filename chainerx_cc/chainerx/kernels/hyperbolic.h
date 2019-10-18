#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

class SinhKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class CoshKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class TanhKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArcsinhKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArccoshKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

}  // namespace chainerx
