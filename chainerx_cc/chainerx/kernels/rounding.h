#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

class CeilKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

class FloorKernel : public Kernel {
public:
    virtual void Call(const Array& x, const Array& out) = 0;
};

}  // namespace chainerx
