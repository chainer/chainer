#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernel_registry.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/native_backend.h"

// Register an kernel statically in NativeBackend.
#define CHAINERX_NATIVE_REGISTER_KERNEL(key_kernel_cls, kernel_cls)                                             \
    static ::chainerx::internal::KernelRegistrar<::chainerx::native::NativeBackend, key_kernel_cls, kernel_cls> \
            s_native_backend_kernel_##kernel_cls{};  // NOLINT(cert-err58-cpp)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(key_kernel_cls, kernel_body, visit_dtype) \
                                                                                                      \
    /* NOLINTNEXTLINE(misc-macro-parentheses,bugprone-macro-parentheses) */                           \
    class Native##key_kernel_cls : public key_kernel_cls {                                            \
    public:                                                                                           \
        void Call(const ::chainerx::Array& x, const ::chainerx::Array& out) override {                \
            ::chainerx::Device& device = x.device();                                                  \
            device.CheckDevicesCompatible(x, out);                                                    \
            const ::chainerx::Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());   \
            visit_dtype(out.dtype(), [&](auto pt) {                                                   \
                using T = typename decltype(pt)::type;                                                \
                struct Impl {                                                                         \
                    void operator()(int64_t i, T x, T& out) {                                         \
                        (void)i;                                                                      \
                        kernel_body                                                                   \
                    }                                                                                 \
                };                                                                                    \
                ::chainerx::native::Elementwise<const T, T>(Impl{}, x_cast, out);                     \
            });                                                                                       \
        }                                                                                             \
    };                                                                                                \
                                                                                                      \
    CHAINERX_NATIVE_REGISTER_KERNEL(key_kernel_cls, Native##key_kernel_cls);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(key_kernel_cls, kernel_body, ::chainerx::VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_UNARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(key_kernel_cls, kernel_body, ::chainerx::VisitDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(key_kernel_cls, kernel_body, visit_dtype)               \
                                                                                                                     \
    /* NOLINTNEXTLINE(misc-macro-parentheses,bugprone-macro-parentheses) */                                          \
    class Native##key_kernel_cls : public key_kernel_cls {                                                           \
    public:                                                                                                          \
        void Call(const ::chainerx::Array& x1, const ::chainerx::Array& x2, const ::chainerx::Array& out) override { \
            ::chainerx::Device& device = x1.device();                                                                \
            device.CheckDevicesCompatible(x1, x2, out);                                                              \
            const ::chainerx::Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());              \
            const ::chainerx::Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());              \
            visit_dtype(out.dtype(), [&](auto pt) {                                                                  \
                using T = typename decltype(pt)::type;                                                               \
                struct Impl {                                                                                        \
                    void operator()(int64_t i, T x1, T x2, T& out) {                                                 \
                        (void)i;                                                                                     \
                        kernel_body                                                                                  \
                    }                                                                                                \
                };                                                                                                   \
                ::chainerx::native::Elementwise<const T, const T, T>(Impl{}, x1_cast, x2_cast, out);                 \
            });                                                                                                      \
        }                                                                                                            \
    };                                                                                                               \
                                                                                                                     \
    CHAINERX_NATIVE_REGISTER_KERNEL(key_kernel_cls, Native##key_kernel_cls);

#define CHAINERX_NATIVE_REGISTER_ELTWISE_FLOAT_BINARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(key_kernel_cls, kernel_body, ::chainerx::VisitFloatingPointDtype)

#define CHAINERX_NATIVE_REGISTER_ELTWISE_BINARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_NATIVE_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(key_kernel_cls, kernel_body, ::chainerx::VisitDtype)
