#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/numeric.cuh"
#include "chainerx/cuda/op_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct SquareImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = x * x; }
};

}  // namespace

void CudaDevice::Square(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SquareImpl<T>{}, x, out);
    });
}

namespace {

template <typename T>
struct SqrtImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Sqrt(x); }
};

}  // namespace

void CudaDevice::Sqrt(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
    CudaSetDeviceScope scope{index()};
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SqrtImpl<T>{}, x_cast, out);
    });
}

namespace {

template <typename T>
struct IsNanImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = cuda::IsNan(x); }
};

}  // namespace

void CudaDevice::IsNan(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(IsNanImpl<T>{}, x, out);
    });
}

namespace {

template <typename T>
struct IsInfImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, bool& out) { out = cuda::IsInf(x); }
};

}  // namespace

void CudaDevice::IsInf(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, bool>(IsInfImpl<T>{}, x, out);
    });
}

template <typename T>
struct CeilImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType x, CudaType& out) { out = cuda::Ceil(x); }
};

class CudaCeilOp : public CeilOp {
public:
    void Call(const Array& x, const Array& out) override {
        Device& device = x.device();
        device.CheckDevicesCompatible(x, out);
        CudaSetDeviceScope scope{device.index()};
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<const T, T>(CeilImpl<T>{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_CUDA(CeilOp, CudaCeilOp);

}  // namespace cuda
}  // namespace chainerx
