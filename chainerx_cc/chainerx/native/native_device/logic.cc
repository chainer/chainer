#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/logic.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/native/reduce.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/logic.h"

namespace chainerx {

namespace internal {
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Equal)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(NotEqual)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Greater)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(GreaterEqual)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LogicalNot)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LogicalAnd)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LogicalOr)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(LogicalXor)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(All)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(Any)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(IsNan)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(IsInf)
CHAINERX_REGISTER_BUILTIN_KEY_KERNEL(IsFinite)
}  // namespace internal

namespace native {
namespace {

class NativeEqualKernel : public EqualKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(EqualKernel, NativeEqualKernel);

class NativeNotEqualKernel : public NotEqualKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 != x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(NotEqualKernel, NativeNotEqualKernel);

class NativeGreaterKernel : public GreaterKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 > x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(GreaterKernel, NativeGreaterKernel);

class NativeGreaterEqualKernel : public GreaterEqualKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 >= x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(GreaterEqualKernel, NativeGreaterEqualKernel);

class NativeLogicalNotKernel : public LogicalNotKernel {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        VisitDtype(x.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x, bool& out) { out = !x; }
            };
            Elementwise<const T, bool>(Impl{}, x, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(LogicalNotKernel, NativeLogicalNotKernel);

class NativeLogicalAndKernel : public LogicalAndKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 && x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(LogicalAndKernel, NativeLogicalAndKernel);

class NativeLogicalOrKernel : public LogicalOrKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 || x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(LogicalOrKernel, NativeLogicalOrKernel);

class NativeLogicalXorKernel : public LogicalXorKernel {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        Dtype dtype = PromoteTypes(x1.dtype(), x2.dtype());
        const Array& x1_cast = x1.dtype() == dtype ? x1 : x1.AsType(dtype);
        const Array& x2_cast = x2.dtype() == dtype ? x2 : x2.AsType(dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = !x1 != !x2; }
            };
            Elementwise<const T, const T, bool>(Impl{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(LogicalXorKernel, NativeLogicalXorKernel);

class NativeAllKernel : public AllKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        a.device().CheckDevicesCompatible(a, out);
        const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
        auto do_all = [&a_cast, &axis, &out](auto in_pt) {
            using In = typename decltype(in_pt)::type;
            struct Impl {
                bool Identity() { return true; }
                bool MapIn(In in, int64_t /*index*/) { return static_cast<bool>(in); }
                void Reduce(bool next, bool& accum) { accum = accum && next; }
                bool MapOut(bool accum) { return accum; }
            };
            Reduce<In, bool>(a_cast, axis, out, Impl{});
        };
        VisitDtype(out.dtype(), do_all);
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AllKernel, NativeAllKernel);

class NativeAnyKernel : public AnyKernel {
public:
    void Call(const Array& a, const Axes& axis, const Array& out) override {
        CHAINERX_ASSERT(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
        a.device().CheckDevicesCompatible(a, out);
        const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
        auto do_any = [&a_cast, &axis, &out](auto in_pt) {
            using In = typename decltype(in_pt)::type;
            struct Impl {
                bool Identity() { return false; }
                bool MapIn(In in, int64_t /*index*/) { return static_cast<bool>(in); }
                void Reduce(bool next, bool& accum) { accum = accum || next; }
                bool MapOut(bool accum) { return accum; }
            };
            Reduce<In, bool>(a_cast, axis, out, Impl{});
        };

        VisitDtype(out.dtype(), do_any);
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(AnyKernel, NativeAnyKernel);

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

}  // namespace
}  // namespace native
}  // namespace chainerx
