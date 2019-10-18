#pragma once

#include <tuple>

#include "chainerx/array.h"
#include "chainerx/kernel.h"
#include "chainerx/routines/linalg.h"

namespace chainerx {

// Matrix multiplication. All the operands are matrices (i.e., two-dimensional arrays).
// Let the shapes of `a` and `b` be `(M, K)` and `(L, N)`, respectively.
// Then, it must hold that `K == L` and the shape of `out` must be `(M, N)`.
// Otherwise, the behavior is undefined.
class DotKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& b, const Array& out) = 0;
};

class SolveKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& b, const Array& out) = 0;
};

class InverseKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& out) = 0;
};

class SvdKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& u, const Array& s, const Array& vt, bool full_matrices, bool compute_uv) = 0;
};

class QrKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& q, const Array& r, const Array& tau, QrMode mode) = 0;
};

class CholeskyKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& out) = 0;
};

class SyevdKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& w, const Array& v, char uplo, bool compute_v) = 0;
};

}  // namespace chainerx
