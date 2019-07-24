#pragma once

#include <tuple>

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

class SolveKernel : public Kernel {
public:
    static const char* name() { return "Solve"; }

    virtual void Call(const Array& a, const Array& b, const Array& out) = 0;
};

class InverseKernel : public Kernel {
public:
    static const char* name() { return "Inverse"; }

    virtual void Call(const Array& a, const Array& out) = 0;
};

class SvdKernel : public Kernel {
public:
    static const char* name() { return "Svd"; }

    virtual void Call(const Array& a, const Array& u, const Array& s, const Array& vt, bool full_matrices) = 0;
};

}  // namespace chainerx
