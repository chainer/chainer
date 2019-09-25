#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/kernel.h"

namespace chainerx {

// Calculate the sum of an array.
// It will be summed over the specified axes.
// `axis` must be normalized so that
// - it has only positive values,
// - it is sorted, and
// - it has no duplicated values.
// Otherwise, the behavior is undefined.
class SumKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

class CumsumKernel : public Kernel {
public:
    virtual void Call(const Array& a, int8_t axis, const Array& out) = 0;
};

class NansumKernel : public Kernel {
public:
    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

}  // namespace chainerx
