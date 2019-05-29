#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/kernel.h"

namespace chainerx {

// Calculates the maximum along specified axes.
// See Sum() for the explanation of arguments.
class AMaxKernel : public Kernel {
public:
    static const char* name() { return "AMax"; }

    virtual void Call(const Array& src, const Axes& axis, const Array& out) = 0;
};

// Calculates the minimum along specified axes.
// See Sum() for the explanation of arguments.
class AMinKernel : public Kernel {
public:
    static const char* name() { return "AMin"; }

    virtual void Call(const Array& src, const Axes& axis, const Array& out) = 0;
};

}  // namespace chainerx
