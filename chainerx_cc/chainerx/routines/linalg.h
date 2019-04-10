#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/op.h"

namespace chainerx {

// Matrix multiplication. All the operands are matrices (i.e., two-dimensional arrays).
// Let the shapes of `a` and `b` be `(M, K)` and `(L, N)`, respectively.
// Then, it must hold that `K == L` and the shape of `out` must be `(M, N)`.
// Otherwise, the behavior is undefined.
class DotOp : public Op {
public:
    static const char* name() { return "Dot"; }

    virtual void Call(const Array& a, const Array& b, const Array& out) = 0;
};

Array Dot(const Array& a, const Array& b, nonstd::optional<Dtype> out_dtype = nonstd::nullopt);

}  // namespace chainerx
