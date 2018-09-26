#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/arithmetic_ops.h"
#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct AddImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = ArithmeticOps<T>::Add(x1, x2); }
};

}  // namespace

// TODO(sonots): support stream
void CudaDevice::Add(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(AddImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct AddASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Add(x1, x2); }
    T x2;
};

}  // namespace

void CudaDevice::AddAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(AddASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct SubtractImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = ArithmeticOps<T>::Subtract(x1, x2); }
};

}  // namespace

void CudaDevice::Subtract(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitNumericDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(SubtractImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct SubtractASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Subtract(x1, x2); }
    T x2;
};

}  // namespace

void CudaDevice::SubtractAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitNumericDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SubtractASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct MultiplyImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = ArithmeticOps<T>::Multiply(x1, x2); }
};

}  // namespace

// TODO(sonots): support stream
void CudaDevice::Multiply(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(MultiplyImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct MultiplyASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Multiply(x1, x2); }
    T x2;
};

}  // namespace

void CudaDevice::MultiplyAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(MultiplyASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct DivideImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = ArithmeticOps<T>::Divide(x1, x2); }
};

}  // namespace

void CudaDevice::Divide(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(DivideImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct DivideASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = ArithmeticOps<T>::Divide(x1, x2); }
    T x2;
};

}  // namespace

void CudaDevice::DivideAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(DivideASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
