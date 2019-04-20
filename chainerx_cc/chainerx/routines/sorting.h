#pragma once

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"

namespace chainerx {

Array ArgMax(const Array& a, const OptionalAxes& axis = nonstd::nullopt);

class ArgMinOp : public Op {
public:
    static const char* name() { return "ArgMin"; }

    virtual Array Call(const Array& a, const OptionalAxes& axis);

protected:
    virtual void Impl(const Array& a, const Axes& axis, const Array& out) = 0;
};

inline Array ArgMin(const Array& a, const OptionalAxes& axis = nonstd::nullopt) { return a.device().backend().CallOp<ArgMinOp>(a, axis); }

}  // namespace chainerx
