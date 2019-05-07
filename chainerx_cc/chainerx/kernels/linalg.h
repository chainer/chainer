#pragma once

#include "chainerx/array.h"
#include "chainerx/kernel.h"

namespace chainerx {

// Matrix multiplication. All the operands are matrices (i.e., two-dimensional arrays).
// Let the shapes of `a` and `b` be `(M, K)` and `(L, N)`, respectively.
// Then, it must hold that `K == L` and the shape of `out` must be `(M, N)`.
// Otherwise, the behavior is undefined.
class DotKernel : public Kernel {
public:
    static const char* name() { return "Dot"; }

    virtual void Call(const Array& a, const Array& b, const Array& out) = 0;
};

}  // namespace chainerx
