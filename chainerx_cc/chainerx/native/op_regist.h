#pragma once

#include "chainerx/native/native_backend.h"
#include "chainerx/op_registry.h"

// Register an op statically in NativeBackend.
#define CHAINERX_NATIVE_REGISTER_OP(key_op_cls, op_cls)                                         \
    static chainerx::internal::OpRegistrar<chainerx::native::NativeBackend, key_op_cls, op_cls> \
            s_native_backend_op_##op_cls{};  // NOLINT(cert-err58-cpp)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_OP(key_op_cls, op_body, visit_dtype) \
    class Native##key_op_cls : public key_op_cls {                                        \
    public:                                                                               \
        void Call(const Array& x, const Array& out) override {                            \
            Device& device = x.device();                                                  \
            device.CheckDevicesCompatible(x, out);                                        \
            const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());   \
            visit_dtype(out.dtype(), [&](auto pt) {                                       \
                using T = typename decltype(pt)::type;                                    \
                struct Impl {                                                             \
                    void operator()(int64_t i, T x, T& out) {                             \
                        (void)i;                                                          \
                        op_body                                                           \
                    }                                                                     \
                };                                                                        \
                Elementwise<const T, T>(Impl{}, x_cast, out);                             \
            });                                                                           \
        }                                                                                 \
    };                                                                                    \
                                                                                          \
    CHAINERX_NATIVE_REGISTER_OP(key_op_cls, Native##key_op_cls);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_OP(key_op_cls, op_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_OP(key_op_cls, op_body, VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_UNARY_OP(key_op_cls, op_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_OP(key_op_cls, op_body, VisitDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_OP(key_op_cls, op_body, visit_dtype)  \
    class Native##key_op_cls : public key_op_cls {                                          \
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
                        op_body                                                             \
                    }                                                                       \
                };                                                                          \
                Elementwise<const T, const T, T>(Impl{}, x1_cast, x2_cast, out);            \
            });                                                                             \
        }                                                                                   \
    };                                                                                      \
                                                                                            \
    CHAINERX_NATIVE_REGISTER_OP(key_op_cls, Native##key_op_cls);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_BINARY_OP(key_op_cls, op_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_OP(key_op_cls, op_body, VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_BINARY_OP(key_op_cls, op_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_OP(key_op_cls, op_body, VisitDtype)
