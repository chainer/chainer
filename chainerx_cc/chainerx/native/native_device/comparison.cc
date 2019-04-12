#include "chainerx/native/native_device.h"

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/native/reduce.h"
#include "chainerx/routines/logic.h"

namespace chainerx {
namespace native {
namespace {

class NativeEqualOp : public EqualOp {
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

CHAINERX_REGISTER_OP_NATIVE(EqualOp, NativeEqualOp);

class NativeNotEqualOp : public NotEqualOp {
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

CHAINERX_REGISTER_OP_NATIVE(NotEqualOp, NativeNotEqualOp);

class NativeGreaterOp : public GreaterOp {
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

CHAINERX_REGISTER_OP_NATIVE(GreaterOp, NativeGreaterOp);

class NativeGreaterEqualOp : public GreaterEqualOp {
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

CHAINERX_REGISTER_OP_NATIVE(GreaterEqualOp, NativeGreaterEqualOp);

class NativeLogicalNotOp : public LogicalNotOp {
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

CHAINERX_REGISTER_OP_NATIVE(LogicalNotOp, NativeLogicalNotOp);

class NativeLogicalAndOp : public LogicalAndOp {
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

CHAINERX_REGISTER_OP_NATIVE(LogicalAndOp, NativeLogicalAndOp);

class NativeLogicalOrOp : public LogicalOrOp {
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

CHAINERX_REGISTER_OP_NATIVE(LogicalOrOp, NativeLogicalOrOp);

class NativeAllOp : public AllOp {
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

CHAINERX_REGISTER_OP_NATIVE(AllOp, NativeAllOp);

class NativeAnyOp : public AnyOp {
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

CHAINERX_REGISTER_OP_NATIVE(AnyOp, NativeAnyOp);

}  // namespace
}  // namespace native
}  // namespace chainerx
