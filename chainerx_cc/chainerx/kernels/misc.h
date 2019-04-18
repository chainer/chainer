#pragma once

// TODO(hvy): Consider moving the content in this file to e.g. kernels/creation.h, in which case this file can be removed.

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

class FillKernel : public Kernel {
public:
    static const char* name() { return "Fill"; }

    virtual void Call(const Array& out, Scalar value) = 0;
};

// Casts the elements from one array to the other dtype, and store into the other.
class AsTypeKernel : public Kernel {
public:
    static const char* name() { return "AsType"; }

    virtual void Call(const Array& a, const Array& out) = 0;
};

}  // namespace chainerx
