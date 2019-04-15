#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/arithmetic_ops.h"
#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/math.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct AddImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = ArithmeticOps<CudaType>::Add(x1, x2); }
};

class CudaAddOp : public AddOp {
public:
    // TODO(sonots): support stream
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, T>(AddImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(AddOp, CudaAddOp);

template <typename T>
struct AddASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Add(x1, x2); }
    CudaType x2;
};

class CudaAddASOp : public AddASOp {
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

CHAINERX_CUDA_REGISTER_OP(AddASOp, CudaAddASOp);

template <typename T>
struct SubtractImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = ArithmeticOps<CudaType>::Subtract(x1, x2); }
};

class CudaSubtractOp : public SubtractOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, T>(SubtractImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(SubtractOp, CudaSubtractOp);

template <typename T>
struct SubtractASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Subtract(x1, x2); }
    CudaType x2;
};

class CudaSubtractASOp : public SubtractASOp {
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

CHAINERX_CUDA_REGISTER_OP(SubtractASOp, CudaSubtractASOp);

template <typename T>
struct MultiplyImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = ArithmeticOps<CudaType>::Multiply(x1, x2); }
};

// TODO(sonots): support stream
class CudaMultiplyOp : public MultiplyOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, T>(MultiplyImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(MultiplyOp, CudaMultiplyOp);

template <typename T>
struct MultiplyASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Multiply(x1, x2); }
    CudaType x2;
};

class CudaMultiplyASOp : public MultiplyASOp {
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

CHAINERX_CUDA_REGISTER_OP(MultiplyASOp, CudaMultiplyASOp);

// CUDA does not have std::div.
__device__ int8_t FloorDivide(int8_t x, int8_t y) { return x / y - ((y >= 0 ? x % y : -(x % y)) < 0 ? 1 : 0); }
__device__ int16_t FloorDivide(int16_t x, int16_t y) { return x / y - ((y >= 0 ? x % y : -(x % y)) < 0 ? 1 : 0); }
__device__ int32_t FloorDivide(int32_t x, int32_t y) { return x / y - ((y >= 0 ? x % y : -(x % y)) < 0 ? 1 : 0); }
__device__ int64_t FloorDivide(int64_t x, int64_t y) { return x / y - ((y >= 0 ? x % y : -(x % y)) < 0 ? 1 : 0); }
__device__ uint8_t FloorDivide(uint8_t x, uint8_t y) { return x / y; }
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

template <typename T>
struct FloorDivideImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = cuda::FloorDivide(x1, x2); }
};

class CudaFloorDivideOp : public FloorDivideOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitNumericDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, T>(FloorDivideImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(FloorDivideOp, CudaFloorDivideOp);

template <typename T>
struct FloorDivideASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = cuda::FloorDivide(x1, x2); }
    CudaType x2;
};

class CudaFloorDivideASOp : public FloorDivideASOp {
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

CHAINERX_CUDA_REGISTER_OP(FloorDivideASOp, CudaFloorDivideASOp);

template <typename T>
struct DivideImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType x2, CudaType& out) { out = ArithmeticOps<CudaType>::Divide(x1, x2); }
};

class CudaDivideOp : public DivideOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& out) override {
        Device& device = x1.device();
        device.CheckDevicesCompatible(x1, x2, out);
        const Array& x1_cast = x1.dtype() == out.dtype() ? x1 : x1.AsType(out.dtype());
        const Array& x2_cast = x2.dtype() == out.dtype() ? x2 : x2.AsType(out.dtype());
        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, const T, T>(DivideImpl<T>{}, x1_cast, x2_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_OP(DivideOp, CudaDivideOp);

template <typename T>
struct DivideASImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x1, CudaType& out) { out = ArithmeticOps<CudaType>::Divide(x1, x2); }
    CudaType x2;
};

class CudaDivideASOp : public DivideASOp {
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

CHAINERX_CUDA_REGISTER_OP(DivideASOp, CudaDivideASOp);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
