#pragma once

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/kernel.h"

namespace chainerx {

class ArgMaxKernel : public Kernel {
public:
    static const char* name() { return "ArgMax"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class ArgMinKernel : public Kernel {
public:
    static const char* name() { return "ArgMin"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class NanArgMaxKernel : public Kernel {
public:
    static const char* name() { return "NanArgMax"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class NanArgMinKernel : public Kernel {
public:
    static const char* name() { return "NanArgMin"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

}  // namespace chainerx
