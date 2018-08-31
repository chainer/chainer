#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct CopyImpl {
    __device__ void operator()(int64_t /*i*/, T a, T& out) { out = a; }
};

}  // namespace

void CudaDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(CopyImpl<T>{}, a, out);
    });
}

namespace {

template <typename InT, typename OutT>
struct AsTypeImpl {
    __device__ void operator()(int64_t /*i*/, InT a, OutT& out) { out = static_cast<OutT>(a); }
};

}  // namespace

void CudaDevice::AsType(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    auto do_astype = [&](auto in_pt, auto out_pt) {
        using InT = typename decltype(in_pt)::type;
        using OutT = typename decltype(out_pt)::type;
        Elementwise<const InT, OutT>(AsTypeImpl<InT, OutT>{}, a, out);
    };
    VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
}

}  // namespace cuda
}  // namespace chainerx
