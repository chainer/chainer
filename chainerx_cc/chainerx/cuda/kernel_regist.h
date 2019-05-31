#pragma once

#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/kernel_registry.h"

// Register an kernel statically in CudaBackend.
#define CHAINERX_CUDA_REGISTER_KERNEL(key_kernel_cls, kernel_cls)                                       \
    static chainerx::internal::KernelRegistrar<chainerx::cuda::CudaBackend, key_kernel_cls, kernel_cls> \
            s_cuda_backend_kernel_##kernel_cls{};

#define CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(key_kernel_cls, kernel_body, visit_dtype) \
                                                                                                    \
    template <typename T>                                                                           \
    struct Cuda##key_kernel_cls##Impl {                                                             \
        using CudaType = cuda_internal::DataType<T>;                                                \
        __device__ void operator()(int64_t i, CudaType x, CudaType& out) {                          \
            (void)i;                                                                                \
            kernel_body                                                                             \
        }                                                                                           \
    };                                                                                              \
                                                                                                    \
    /* NOLINTNEXTLINE(misc-macro-parentheses) */                                                    \
    class Cuda##key_kernel_cls : public key_kernel_cls {                                            \
    public:                                                                                         \
        void Call(const Array& x, const Array& out) override {                                      \
            Device& device = x.device();                                                            \
            device.CheckDevicesCompatible(x, out);                                                  \
            CudaSetDeviceScope scope{device.index()};                                               \
            const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());             \
            visit_dtype(out.dtype(), [&](auto pt) {                                                 \
                using T = typename decltype(pt)::type;                                              \
                Elementwise<const T, T>(Cuda##key_kernel_cls##Impl<T>{}, x_cast, out);              \
            });                                                                                     \
        }                                                                                           \
    };                                                                                              \
                                                                                                    \
    CHAINERX_CUDA_REGISTER_KERNEL(key_kernel_cls, Cuda##key_kernel_cls)

#define CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_UNARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(key_kernel_cls, kernel_body, VisitFloatingPointDtype)

#define CHAINERX_CUDA_REGISTER_ELTWISE_UNARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_UNARY_KERNEL(key_kernel_cls, kernel_body, VisitDtype)

#define CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(key_kernel_cls, kernel_body, visit_dtype)      \
                                                                                                          \
    template <typename T>                                                                                 \
    struct Cuda##key_kernel_cls##Impl {                                                                   \
        using CudaType = cuda_internal::DataType<T>;                                                      \
        __device__ void operator()(int64_t i, CudaType x1, CudaType x2, CudaType& out) {                  \
            (void)i;                                                                                      \
            kernel_body                                                                                   \
        }                                                                                                 \
    };                                                                                                    \
                                                                                                          \
    /* NOLINTNEXTLINE(misc-macro-parentheses) */                                                          \
    class Cuda##key_kernel_cls : public key_kernel_cls {                                                  \
    public:                                                                                               \
        void Call(const Array& x1, const Array& x2, const Array& out) override {                          \
            Device& device = x1.device();                                                                 \
            device.CheckDevicesCompatible(x1, x2, out);                                                   \
            const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());               \
            const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());               \
            CudaSetDeviceScope scope{device.index()};                                                     \
            visit_dtype(out.dtype(), [&](auto pt) {                                                       \
                using T = typename decltype(pt)::type;                                                    \
                Elementwise<const T, const T, T>(Cuda##key_kernel_cls##Impl<T>{}, x1_cast, x2_cast, out); \
            });                                                                                           \
        }                                                                                                 \
    };                                                                                                    \
                                                                                                          \
    CHAINERX_CUDA_REGISTER_KERNEL(key_kernel_cls, Cuda##key_kernel_cls);

#define CHAINERX_CUDA_REGISTER_ELTWISE_FLOAT_BINARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(key_kernel_cls, kernel_body, VisitFloatingPointDtype)

#define CHAINERX_CUDA_REGISTER_ELTWISE_BINARY_KERNEL(key_kernel_cls, kernel_body) \
    CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(key_kernel_cls, kernel_body, VisitDtype)
