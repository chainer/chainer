#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/math.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/type_util.h"

namespace chainerx {
namespace native {
namespace {

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(SqrtKernel, { out = chainerx::Sqrt(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(SquareKernel, { out = x * x; }, VisitNumericDtype);

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(FabsKernel, { out = chainerx::Fabs(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(SignKernel, { out = chainerx::Sign(x); }, VisitNumericDtype);

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(CeilKernel, { out = chainerx::Ceil(x); });

CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(FloorKernel, { out = chainerx::Floor(x); });

class NativeIsNanKernel : public IsNanKernel {
public:
    void Call(const Array& x, const Array& out) override {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x, bool& out) { out = chainerx::IsNan(x); }
            };
            Elementwise<const T, bool>(Impl{}, x, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(IsNanKernel, NativeIsNanKernel);

class NativeIsInfKernel : public IsInfKernel {
public:
    void Call(const Array& x, const Array& out) override {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x, bool& out) { out = chainerx::IsInf(x); }
            };
            Elementwise<const T, bool>(Impl{}, x, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(IsInfKernel, NativeIsInfKernel);

class NativeIsFiniteKernel : public IsFiniteKernel {
public:
    void Call(const Array& x, const Array& out) override {
        x.device().CheckDevicesCompatible(x, out);
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x, bool& out) { out = !(chainerx::IsInf(x) || chainerx::IsNan(x)); }
            };
            Elementwise<const T, bool>(Impl{}, x, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(IsFiniteKernel, NativeIsFiniteKernel);

class NativeIfLessElseASSAKernel : public IfLessElseASSAKernel {
public:
    void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override {
        x1.device().CheckDevicesCompatible(x1, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        VisitNumericDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            VisitNumericDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                struct Impl {
                    void operator()(int64_t /*i*/, In x1, Out neg, Out& out) { out = x1 < x2 ? pos : neg; }
                    In x2;
                    Out pos;
                };
                Elementwise<const In, const Out, Out>(Impl{static_cast<In>(x2), static_cast<Out>(pos)}, x1_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(IfLessElseASSAKernel, NativeIfLessElseASSAKernel);

class NativeIfGreaterElseASSAKernel : public IfGreaterElseASSAKernel {
public:
    void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override {
        x1.device().CheckDevicesCompatible(x1, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        VisitNumericDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            VisitNumericDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                struct Impl {
                    void operator()(int64_t /*i*/, In x1, Out neg, Out& out) { out = x1 > x2 ? pos : neg; }
                    In x2;
                    Out pos;
                };
                Elementwise<const In, const Out, Out>(Impl{static_cast<In>(x2), static_cast<Out>(pos)}, x1_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(IfGreaterElseASSAKernel, NativeIfGreaterElseASSAKernel);

class NativeIfGreaterElseAAAAKernel : public IfGreaterElseAAAAKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) override {
        x1.device().CheckDevicesCompatible(x1, x2, pos, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& x2_cast = x2.dtype() == x_dtype ? x2 : x2.AsType(x_dtype);
        const Array& pos_cast = pos.dtype() == out.dtype() ? pos : pos.AsType(out.dtype());
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        VisitNumericDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            VisitNumericDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                struct Impl {
                    void operator()(int64_t /*i*/, In x1, In x2, Out pos, Out neg, Out& out) { out = x1 > x2 ? pos : neg; }
                };
                Elementwise<const In, const In, const Out, const Out, Out>(Impl{}, x1_cast, x2_cast, pos_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(IfGreaterElseAAAAKernel, NativeIfGreaterElseAAAAKernel);

}  // namespace
}  // namespace native
}  // namespace chainerx
