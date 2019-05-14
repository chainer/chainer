#include "chainerx/routines/binary.h"

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/binary.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {

template <typename Impl>
inline Array BitwiseImpl(Impl&& impl, const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) == DtypeKind::kFloat || GetKind(x2.dtype()) == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }
    Dtype out_dtype = ResultType(x1, x2);
    return internal::BroadcastBinary(impl, x1, x2, out_dtype);
}

template <typename Kernel>
inline void ApplyBitwiseImpl(const Array& x1, const Array& x2, const Array& out) {
    NoBackpropModeScope scope;
    CheckEqual(x1.shape(), x2.shape());
    x1.device().backend().CallKernel<Kernel>(x1, x2, out);
}

template <typename Kernel>
inline void ApplyBitwiseASImpl(const Array& x1, Scalar x2, const Array& out) {
    NoBackpropModeScope scope;
    if (GetKind(x1.dtype()) == DtypeKind::kFloat || x2.kind() == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }
    x1.device().backend().CallKernel<Kernel>(x1, x2, out);
}

namespace internal {

template <typename Impl>
inline void IBitwiseImpl(Impl&& impl, const Array& x1, const Array& x2) {
    if (GetKind(x1.dtype()) == DtypeKind::kFloat || GetKind(x2.dtype()) == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }
    if (GetKind(x1.dtype()) == DtypeKind::kBool && GetKind(x2.dtype()) != DtypeKind::kBool) {
        throw DtypeError{"Bitwise operations don't allow updating Bool array with integer array"};
    }
    internal::BroadcastBinaryInPlace(impl, x1, x2);
}

void IBitwiseAnd(const Array& x1, const Array& x2) { IBitwiseImpl(ApplyBitwiseImpl<BitwiseAndKernel>, x1, x2); }

void IBitwiseAnd(const Array& x1, Scalar x2) { internal::BinaryInPlace(ApplyBitwiseASImpl<BitwiseAndASKernel>, x1, x2); }

void IBitwiseOr(const Array& x1, const Array& x2) { IBitwiseImpl(ApplyBitwiseImpl<BitwiseOrKernel>, x1, x2); }

void IBitwiseOr(const Array& x1, Scalar x2) { internal::BinaryInPlace(ApplyBitwiseASImpl<BitwiseOrASKernel>, x1, x2); }

void IBitwiseXor(const Array& x1, const Array& x2) { IBitwiseImpl(ApplyBitwiseImpl<BitwiseXorKernel>, x1, x2); }

void IBitwiseXor(const Array& x1, Scalar x2) { internal::BinaryInPlace(ApplyBitwiseASImpl<BitwiseXorASKernel>, x1, x2); }

}  // namespace internal

Array BitwiseAnd(const Array& x1, const Array& x2) { return BitwiseImpl(ApplyBitwiseImpl<BitwiseAndKernel>, x1, x2); }

Array BitwiseAnd(const Array& x1, Scalar x2) {
    return internal::Binary(ApplyBitwiseASImpl<BitwiseAndASKernel>, x1, x2, ResultType(x1, x2, true));
}

Array BitwiseAnd(Scalar x1, const Array& x2) { return BitwiseAnd(x2, x1); }

Array BitwiseOr(const Array& x1, const Array& x2) { return BitwiseImpl(ApplyBitwiseImpl<BitwiseOrKernel>, x1, x2); }

Array BitwiseOr(const Array& x1, Scalar x2) {
    return internal::Binary(ApplyBitwiseASImpl<BitwiseOrASKernel>, x1, x2, ResultType(x1, x2, true));
}

Array BitwiseOr(Scalar x1, const Array& x2) { return BitwiseOr(x2, x1); }

Array BitwiseXor(const Array& x1, const Array& x2) { return BitwiseImpl(ApplyBitwiseImpl<BitwiseXorKernel>, x1, x2); }

Array BitwiseXor(const Array& x1, Scalar x2) {
    return internal::Binary(ApplyBitwiseASImpl<BitwiseXorASKernel>, x1, x2, ResultType(x1, x2, true));
}

Array BitwiseXor(Scalar x1, const Array& x2) { return BitwiseXor(x2, x1); }

}  // namespace chainerx
