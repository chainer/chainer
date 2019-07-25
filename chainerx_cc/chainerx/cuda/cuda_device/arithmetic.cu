#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/arithmetic_ops.h"
#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/arithmetic.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

CHAINERX_CUDA_REGISTER_ELTWISE_BINARY_KERNEL(AddKernel, { out = ArithmeticOps<CudaType>::Add(x1, x2); });

template <typename T>
struct AddASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Add(x1, x2); }
    CudaType x2;
};

class CudaAddASKernel : public AddASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(AddASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(AddASKernel, CudaAddASKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(SubtractKernel, { out = ArithmeticOps<CudaType>::Subtract(x1, x2); }, VisitNumericDtype);

template <typename T>
struct SubtractASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Subtract(x1, x2); }
    CudaType x2;
};

class CudaSubtractASKernel : public SubtractASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(SubtractASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SubtractASKernel, CudaSubtractASKernel);

// TODO(sonots): support stream
CHAINERX_CUDA_REGISTER_ELTWISE_BINARY_KERNEL(MultiplyKernel, { out = ArithmeticOps<CudaType>::Multiply(x1, x2); });

template <typename T>
struct MultiplyASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Multiply(x1, x2); }
    CudaType x2;
};

class CudaMultiplyASKernel : public MultiplyASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(MultiplyASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(MultiplyASKernel, CudaMultiplyASKernel);

// CUDA does not have std::div, which is used for the native backend.
template <typename T>
__device__ T FloorDivideImpl(T x, T y) {
    if (y == 0) {
        return 0;
    }
    return x / y - ((y >= 0 ? x % y : -(x % y)) < 0 ? 1 : 0);
}
__device__ int8_t FloorDivide(int8_t x, int8_t y) { return FloorDivideImpl(x, y); }
__device__ int16_t FloorDivide(int16_t x, int16_t y) { return FloorDivideImpl(x, y); }
__device__ int32_t FloorDivide(int32_t x, int32_t y) { return FloorDivideImpl(x, y); }
__device__ int64_t FloorDivide(int64_t x, int64_t y) { return FloorDivideImpl(x, y); }
__device__ uint8_t FloorDivide(uint8_t x, uint8_t y) {
    if (y == 0) {
        return 0;
    }
    return x / y;
}
__device__ float FloorDivide(float x, float y) {
    float rem = std::fmod(x, y);
    return (x - rem) / y - ((rem < 0 && y > 0) || (rem > 0 && y < 0) ? 1 : 0);
}
__device__ double FloorDivide(double x, double y) {
    double rem = std::fmod(x, y);
    return (x - rem) / y - ((rem < 0 && y > 0) || (rem > 0 && y < 0) ? 1 : 0);
}
__device__ cuda::Float16 FloorDivide(cuda::Float16 x, cuda::Float16 y) {
    return cuda::Float16{FloorDivide(static_cast<float>(x), static_cast<float>(y))};
}

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(FloorDivideKernel, { out = cuda::FloorDivide(x1, x2); }, VisitNumericDtype);

template <typename T>
struct FloorDivideASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = cuda::FloorDivide(x1, x2); }
    CudaType x2;
};

class CudaFloorDivideASKernel : public FloorDivideASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(FloorDivideASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(FloorDivideASKernel, CudaFloorDivideASKernel);

template <typename T>
struct FloorDivideSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = cuda::FloorDivide(x1, x2); }
    CudaType x1;
};

class CudaFloorDivideSAKernel : public FloorDivideSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(FloorDivideSAImpl<T>{static_cast<CudaType>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(FloorDivideSAKernel, CudaFloorDivideSAKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_BINARY_KERNEL(DivideKernel, { out = ArithmeticOps<CudaType>::Divide(x1, x2); });

template <typename T>
struct DivideASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Divide(x1, x2); }
    CudaType x2;
};

class CudaDivideASKernel : public DivideASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(DivideASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(DivideASKernel, CudaDivideASKernel);

template <typename T>
struct DivideSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = ArithmeticOps<CudaType>::Divide(x1, x2); }
    CudaType x1;
};

class CudaDivideSAKernel : public DivideSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(DivideSAImpl<T>{static_cast<CudaType>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(DivideSAKernel, CudaDivideSAKernel);

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(PowerKernel, { out = cuda::Power(x1, x2); }, VisitNumericDtype);

template <typename T>
struct PowerASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = cuda::Power(x1, x2); }
    CudaType x2;
};

class CudaPowerASKernel : public PowerASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(PowerASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(PowerASKernel, CudaPowerASKernel);

template <typename T>
struct PowerSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = cuda::Power(x1, x2); }
    CudaType x1;
};

class CudaPowerSAKernel : public PowerSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(PowerSAImpl<T>{static_cast<CudaType>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(PowerSAKernel, CudaPowerSAKernel);

// CUDA does not have std::mod, which is used for the native backend.
template <typename T>
__device__ T ModSignedIntegerImpl(T x, T y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    if (x < 0) {
        if (y > 0) {
            T z = (-x) % y;
            return z == 0 ? 0 : y - z;
        }
        return -(-x % (-y));
    }
    if (y < 0) {
        T z = x % (-y);
        return z == 0 ? 0 : y - z;
    }
    return x % y;
}
__device__ int8_t Mod(int8_t x, int8_t y) { return ModSignedIntegerImpl(x, y); }
__device__ int16_t Mod(int16_t x, int16_t y) { return ModSignedIntegerImpl(x, y); }
__device__ int32_t Mod(int32_t x, int32_t y) { return ModSignedIntegerImpl(x, y); }
__device__ int64_t Mod(int64_t x, int64_t y) { return ModSignedIntegerImpl(x, y); }
__device__ uint8_t Mod(uint8_t x, uint8_t y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    return x % y;
}
__device__ T ModFloatImpl(T x, T y) {
    if (x == 0 || y == 0) {
        return 0;
    }
    if (x < 0) {
        if (y > 0) {
            T z = std::fmod(-x, y);
            return z == 0 ? 0 : y - z;
        }
        return -std::fmod(-x, -y);
    }
    if (y < 0) {
        T x = std::fmod(x, -y);
        return z == 0 ? 0 : y - z;
    }
    return std::fmod(x, y);
}
__device__ double Mod(double x, double y) { return ModFloatImpl(x, y); }
__device__ float Mod(float x, float y) { return ModFloatImpl(x, y); }
__device__ cuda::Float16 Mod(cuda::Float16 x, cuda::Float16 y) { return cuda::Float16{Mod(static_cast<float>(x), static_cast<float>(y))}; }

CHAINERX_CUDA_REGISTER_ELTWISE_DTYPE_BINARY_KERNEL(ModAAKernel, { out = cuda::Mod(x1, x2); }, VisitNumericDtype);

template <typename T>
struct ModASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = cuda::Mod(x1, x2); }
    CudaType x2;
};

class CudaModASKernel : public ModASKernel {
public:
    void Call(const Array& x1, Scalar x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(ModASImpl<T>{static_cast<CudaType>(x2)}, x1_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ModASKernel, CudaModASKernel);

template <typename T>
struct ModSAImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x2, CudaType& out) { out = cuda::Mod(x1, x2); }
    CudaType x1;
};

class CudaModSAKernel : public ModSAKernel {
public:
    void Call(Scalar x1, const Array& x2, const Array& out) override {
        Device& device = x2.device();
        device.CheckDevicesCompatible(x2, out);
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const T, T>(ModSAImpl<T>{static_cast<CudaType>(x1)}, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(ModSAKernel, CudaModSAKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
