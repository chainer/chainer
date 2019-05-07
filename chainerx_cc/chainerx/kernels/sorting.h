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

}  // namespace chainerx
