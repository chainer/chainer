#pragma once

#include "chainerx/array.h"
#include "chainerx/op.h"
#include "chainerx/scalar.h"

namespace chainerx {

class FillOp : public Op {
public:
    static const char* name() { return "Fill"; }

    virtual void Call(const Array& out, Scalar value) = 0;
};

// Casts the elements from one array to the other dtype, and store into the other.
class AsTypeOp : public Op {
public:
    static const char* name() { return "AsType"; }

    virtual void Call(const Array& a, const Array& out) = 0;
};

}  // namespace chainerx
