#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/kernel.h"
#include "chainerx/routines/indexing.h"

namespace chainerx {

class AddAtKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out, IndexBoundsMode mode) = 0;
};

class TakeKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& out, IndexBoundsMode mode) = 0;
};

class WhereKernel : public Kernel {
public:
    virtual void Call(const Array& condition, const Array& x, const Array& y, const Array& out) = 0;
};

class WhereAASKernel : public Kernel {
public:
    virtual void Call(const Array& condition, const Array& x, Scalar y, const Array& out) = 0;
};

class WhereASAKernel : public Kernel {
public:
    virtual void Call(const Array& condition, Scalar x, const Array& y, const Array& out) = 0;
};

class WhereASSKernel : public Kernel {
public:
    virtual void Call(const Array& condition, Scalar x, Scalar y, const Array& out) = 0;
};

}  // namespace chainerx
