#include "chainerx/routines/logic.h"

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
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

}  // namespace

Array Equal(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    auto func = [](const Array& x1, const Array& x2, Array& out) { return x1.device().Equal(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array NotEqual(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    auto func = [](const Array& x1, const Array& x2, Array& out) { return x1.device().NotEqual(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array Greater(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    auto func = [](const Array& x1, const Array& x2, Array& out) { return x1.device().Greater(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array GreaterEqual(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    auto func = [](const Array& x1, const Array& x2, Array& out) { return x1.device().GreaterEqual(x1, x2, out); };
    return BroadcastComparison(func, x1, x2);
}

Array Less(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    auto func = [](const Array& x1, const Array& x2, Array& out) { return x1.device().Greater(x2, x1, out); };
    return BroadcastComparison(func, x1, x2);
}

Array LessEqual(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    auto func = [](const Array& x1, const Array& x2, Array& out) { return x1.device().GreaterEqual(x2, x1, out); };
    return BroadcastComparison(func, x1, x2);
}

Array LogicalNot(const Array& x1) {
    Array out = Empty(x1.shape(), Dtype::kBool, x1.device());
    {
        NoBackpropModeScope scope{};
        x1.device().LogicalNot(x1, out);
    }
    return out;
}

}  // namespace chainerx
