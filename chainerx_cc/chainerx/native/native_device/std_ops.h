#pragma once

#include "chainerx/native/op_regist.h"

#define CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_OP(func, func_def, visit_dtype)    \
    class Native##func##Op : public func##Op {                                          \
    public:                                                                             \
        void Call(const Array& x, const Array& out) override {                          \
            Device& device = x.device();                                                \
            device.CheckDevicesCompatible(x, out);                                      \
            const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype()); \
            visit_dtype(out.dtype(), [&](auto pt) {                                     \
                using T = typename decltype(pt)::type;                                  \
                struct Impl {                                                           \
                    void operator()(int64_t i, T x, T& out) {                           \
                        (void)i;                                                        \
                        func_def                                                        \
                    }                                                                   \
                };                                                                      \
                Elementwise<const T, T>(Impl{}, x_cast, out);                           \
            });                                                                         \
        }                                                                               \
    };                                                                                  \
                                                                                        \
    CHAINERX_REGISTER_OP_NATIVE(func##Op, Native##func##Op);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(func, func_def) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_OP(func, func_def, VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_UNARY_OP(func, func_def) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_OP(func, func_def, VisitDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_OP(func, func_def, visit_dtype)       \
    class Native##func##Op : public func##Op {                                              \
    public:                                                                                 \
        void Call(const Array& x1, const Array& x2, const Array& out) override {            \
            Device& device = x1.device();                                                   \
            device.CheckDevicesCompatible(x1, x2, out);                                     \
            const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype()); \
            const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype()); \
            visit_dtype(out.dtype(), [&](auto pt) {                                         \
                using T = typename decltype(pt)::type;                                      \
                struct Impl {                                                               \
                    void operator()(int64_t i, T x1, T x2, T& out) {                        \
                        (void)i;                                                            \
                        func_def                                                            \
                    }                                                                       \
                };                                                                          \
                Elementwise<const T, const T, T>(Impl{}, x1_cast, x2_cast, out);            \
            });                                                                             \
        }                                                                                   \
    };                                                                                      \
                                                                                            \
    CHAINERX_REGISTER_OP_NATIVE(func##Op, Native##func##Op);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_BINARY_OP(func, func_def) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_OP(func, func_def, VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_BINARY_OP(func, func_def) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_OP(func, func_def, VisitDtype)
