#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class CeilKernel : public Kernel {
public:
    static const char* name() { return "Ceil"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class FloorKernel : public Kernel {
public:
    static const char* name() { return "Floor"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

}  // namespace chainerx
