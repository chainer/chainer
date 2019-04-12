#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/device.h"
#include "chainerx/op.h"

namespace chainerx {

class ArgMaxOp : public Op {
public:
    static const char* name() { return "ArgMax"; }

    virtual void Call(const Array& a, const Axes& axis, const Array& out) = 0;
};

Array ArgMax(const Array& a, const OptionalAxes& axis = nonstd::nullopt);

}  // namespace chainerx
