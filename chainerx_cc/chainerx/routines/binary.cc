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
namespace {

void CheckBitwiseDtypes(DtypeKind kind1, DtypeKind kind2) {
    if (kind1 == DtypeKind::kFloat || kind2 == DtypeKind::kFloat) {
        throw DtypeError{"Bitwise operations don't support Float types"};
    }
}

void CheckInplaceBitwiseDtypes(DtypeKind out_kind, DtypeKind in_kind) {
    CheckBitwiseDtypes(out_kind, in_kind);
    if (out_kind == DtypeKind::kBool && in_kind != DtypeKind::kBool) {
        throw DtypeError{"Bitwise operations don't allow updating Bool array with integer array"};
    }
}

void CheckBitwiseDtypes(const Array& x1, const Array& x2) { CheckBitwiseDtypes(GetKind(x1.dtype()), GetKind(x2.dtype())); }

void CheckBitwiseDtypes(const Array& x1, const Scalar& x2) { CheckBitwiseDtypes(GetKind(x1.dtype()), x2.kind()); }

void CheckInplaceBitwiseDtypes(const Array& x1, const Array& x2) { CheckInplaceBitwiseDtypes(GetKind(x1.dtype()), GetKind(x2.dtype())); }

void CheckInplaceBitwiseDtypes(const Array& x1, const Scalar& x2) { CheckInplaceBitwiseDtypes(GetKind(x1.dtype()), x2.kind()); }

template <typename Impl>
void BitwiseImpl(const Array& x1, const Array& x2, const Array& out) {
    CHAINERX_ASSERT(x1.shape() == x2.shape());
    NoBackpropModeScope scope{};
    x1.device().backend().CallKernel<Impl>(x1, x2, out);
}

template <typename Impl>
void BitwiseASImpl(const Array& x1, Scalar x2, const Array& out) {
    NoBackpropModeScope scope{};
    x1.device().backend().CallKernel<Impl>(x1, x2, out);
}

}  // namespace

namespace internal {

void IBitwiseAnd(const Array& x1, const Array& x2) {
    CheckInplaceBitwiseDtypes(x1, x2);
    internal::BroadcastBinaryInplace(BitwiseImpl<BitwiseAndKernel>, x1, x2);
}

void IBitwiseAnd(const Array& x1, Scalar x2) {
    CheckInplaceBitwiseDtypes(x1, x2);
    internal::BinaryInplace(BitwiseASImpl<BitwiseAndASKernel>, x1, x2);
}

void IBitwiseOr(const Array& x1, const Array& x2) {
    CheckInplaceBitwiseDtypes(x1, x2);
    internal::BroadcastBinaryInplace(BitwiseImpl<BitwiseOrKernel>, x1, x2);
}

void IBitwiseOr(const Array& x1, Scalar x2) {
    CheckInplaceBitwiseDtypes(x1, x2);
    internal::BinaryInplace(BitwiseASImpl<BitwiseOrASKernel>, x1, x2);
}

void IBitwiseXor(const Array& x1, const Array& x2) {
    CheckInplaceBitwiseDtypes(x1, x2);
    internal::BroadcastBinaryInplace(BitwiseImpl<BitwiseXorKernel>, x1, x2);
}

void IBitwiseXor(const Array& x1, Scalar x2) {
    CheckInplaceBitwiseDtypes(x1, x2);
    internal::BinaryInplace(BitwiseASImpl<BitwiseXorASKernel>, x1, x2);
}

}  // namespace internal

Array BitwiseAnd(const Array& x1, const Array& x2) {
    CheckBitwiseDtypes(x1, x2);
    return internal::BroadcastBinary(BitwiseImpl<BitwiseAndKernel>, x1, x2, ResultType(x1, x2));
}

Array BitwiseAnd(const Array& x1, Scalar x2) {
    CheckBitwiseDtypes(x1, x2);
    return internal::Binary(BitwiseASImpl<BitwiseAndASKernel>, x1, x2, ResultType(x1, x2));
}

Array BitwiseAnd(Scalar x1, const Array& x2) { return BitwiseAnd(x2, x1); }

Array BitwiseOr(const Array& x1, const Array& x2) {
    CheckBitwiseDtypes(x1, x2);
    return internal::BroadcastBinary(BitwiseImpl<BitwiseOrKernel>, x1, x2, ResultType(x1, x2));
}

Array BitwiseOr(const Array& x1, Scalar x2) {
    CheckBitwiseDtypes(x1, x2);
    return internal::Binary(BitwiseASImpl<BitwiseOrASKernel>, x1, x2, ResultType(x1, x2));
}

Array BitwiseOr(Scalar x1, const Array& x2) { return BitwiseOr(x2, x1); }

Array BitwiseXor(const Array& x1, const Array& x2) {
    CheckBitwiseDtypes(x1, x2);
    return internal::BroadcastBinary(BitwiseImpl<BitwiseXorKernel>, x1, x2, ResultType(x1, x2));
}

Array BitwiseXor(const Array& x1, Scalar x2) {
    CheckBitwiseDtypes(x1, x2);
    return internal::Binary(BitwiseASImpl<BitwiseXorASKernel>, x1, x2, ResultType(x1, x2));
}

Array BitwiseXor(Scalar x1, const Array& x2) { return BitwiseXor(x2, x1); }

}  // namespace chainerx
