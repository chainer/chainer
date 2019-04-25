#include "chainerx/routines/logic.h"

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

namespace {

template <typename Impl>
Array BroadcastComparison(Impl&& impl, const Array& x1, const Array& x2) {
    auto func = [&impl](const Array& x1, const Array& x2) {
        Array out = Empty(x1.shape(), Dtype::kBool, x1.device());
        {
            NoBackpropModeScope scope{};
            impl(x1, x2, out);
        }
        return out;
    };

    if (x1.shape() == x2.shape()) {
        return func(x1, x2);
    }
    Shape result_shape = internal::BroadcastShapes(x1.shape(), x2.shape());
    if (x1.shape() == result_shape) {
        return func(x1, x2.BroadcastTo(result_shape));
    }
    if (x2.shape() == result_shape) {
        return func(x1.BroadcastTo(result_shape), x2);
    }
    return func(x1.BroadcastTo(result_shape), x2.BroadcastTo(result_shape));
}

void CheckLogicDtypes(const Array& x1, const Array& x2) {
    if ((x1.dtype() == Dtype::kBool) != (x2.dtype() == Dtype::kBool)) {
        throw DtypeError{"Comparison of ", GetDtypeName(x1.dtype()), " and ", GetDtypeName(x2.dtype()), " is not supported."};
    }
}

}  // namespace

Array EqualOp::Call(const Array& x1, const Array& x2) {
    CheckLogicDtypes(x1, x2);
    auto func = [this](const Array& x1, const Array& x2, Array& out) { Impl(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array NotEqualOp::Call(const Array& x1, const Array& x2) {
    CheckLogicDtypes(x1, x2);
    auto func = [this](const Array& x1, const Array& x2, Array& out) { Impl(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array GreaterOp::Call(const Array& x1, const Array& x2) {
    CheckLogicDtypes(x1, x2);
    auto func = [this](const Array& x1, const Array& x2, Array& out) { Impl(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array GreaterEqualOp::Call(const Array& x1, const Array& x2) {
    CheckLogicDtypes(x1, x2);
    auto func = [this](const Array& x1, const Array& x2, Array& out) { return Impl(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array LogicalNotOp::Call(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        Impl(x, out);
    }
    return out;
}

}  // namespace chainerx
