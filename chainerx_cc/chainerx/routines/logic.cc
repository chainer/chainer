#include "chainerx/routines/logic.h"

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/logic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"

namespace chainerx {

namespace {

template <typename KernelType>
class LogicBinaryImpl {
public:
    LogicBinaryImpl() = default;
    void operator()(const Array& x1, const Array& x2, Array& out) const {
        NoBackpropModeScope scope{};
        x1.device().backend().CallKernel<KernelType>(x1, x2, out);
    }
};

void CheckLogicDtypes(DtypeKind kind1, DtypeKind kind2) {
    if ((kind1 == DtypeKind::kBool) != (kind2 == DtypeKind::kBool)) {
        throw DtypeError{"Comparison of bool and non-bool dtypes is not supported."};
    }
}

void CheckLogicDtypes(const Array& x1, const Array& x2) { return CheckLogicDtypes(GetKind(x1.dtype()), GetKind(x2.dtype())); }

void CheckLogicDtypes(const Array& x1, Scalar x2) { return CheckLogicDtypes(GetKind(x1.dtype()), x2.kind()); }

template <typename KernelType>
Array LogicBinary(const Array& x1, const Array& x2) {
    CheckLogicDtypes(x1, x2);
    return internal::BroadcastBinary(LogicBinaryImpl<KernelType>{}, x1, x2, Dtype::kBool);
}

}  // namespace

Array Equal(const Array& x1, const Array& x2) { return LogicBinary<EqualKernel>(x1, x2); }

Array NotEqual(const Array& x1, const Array& x2) { return LogicBinary<NotEqualKernel>(x1, x2); }

Array Greater(const Array& x1, const Array& x2) { return LogicBinary<GreaterKernel>(x1, x2); }

Array GreaterEqual(const Array& x1, const Array& x2) { return LogicBinary<GreaterEqualKernel>(x1, x2); }

Array LogicalNot(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<LogicalNotKernel>(x, out);
    }
    return out;
}

Array LogicalAnd(const Array& x1, const Array& x2) { return LogicBinary<LogicalAndKernel>(x1, x2); }

Array LogicalAnd(const Array& x1, Scalar x2) {
    CheckLogicDtypes(x1, x2);
    return static_cast<bool>(x2) ? x1.AsType(Dtype::kBool) : Zeros(x1.shape(), Dtype::kBool, x1.device());
}

Array LogicalOr(const Array& x1, const Array& x2) { return LogicBinary<LogicalOrKernel>(x1, x2); }

Array LogicalOr(const Array& x1, Scalar x2) {
    CheckLogicDtypes(x1, x2);
    return static_cast<bool>(x2) ? Ones(x1.shape(), Dtype::kBool, x1.device()) : x1.AsType(Dtype::kBool);
}

Array LogicalXor(const Array& x1, const Array& x2) { return LogicBinary<LogicalXorKernel>(x1, x2); }

Array All(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::EmptyReduced(a.shape(), Dtype::kBool, sorted_axis, keepdims, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<AllKernel>(a, sorted_axis, out);
    }
    return out;
}

Array Any(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::EmptyReduced(a.shape(), Dtype::kBool, sorted_axis, keepdims, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<AnyKernel>(a, sorted_axis, out);
    }
    return out;
}

Array IsNan(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsNanKernel>(x, out);
    }
    return out;
}

Array IsInf(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsInfKernel>(x, out);
    }
    return out;
}

Array IsFinite(const Array& x) {
    Array out = Empty(x.shape(), Dtype::kBool, x.device());
    {
        NoBackpropModeScope scope{};
        x.device().backend().CallKernel<IsFiniteKernel>(x, out);
    }
    return out;
}

}  // namespace chainerx
