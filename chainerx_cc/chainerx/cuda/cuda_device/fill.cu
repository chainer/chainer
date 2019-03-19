#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <mutex>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cast.cuh"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
struct ArangeImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t i, CudaType& out) { out = start + step * static_cast<CudaType>(i); }
    CudaType start;
    CudaType step;
};

}  // namespace

void CudaDevice::Arange(Scalar start, Scalar step, const Array& out) {
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using CudaType = cuda_internal::DataType<T>;
        Elementwise<T>(ArangeImpl<T>{static_cast<CudaType>(start), static_cast<CudaType>(step)}, out);
    });
}

namespace {

template <typename T>
struct FillImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, CudaType& out) { out = value; }
    CudaType value;
};

}  // namespace

void CudaDevice::Fill(const Array& out, Scalar value) {
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        using CudaType = cuda_internal::DataType<T>;
        Elementwise<T>(FillImpl<T>{static_cast<CudaType>(value)}, out);
    });
}

namespace {

template <typename T>
struct IdentityImpl {
    using CudaType = cuda_internal::DataType<T>;
    explicit IdentityImpl(int64_t n) : n_plus_one{n + 1} {}
    __device__ void operator()(int64_t i, CudaType& out) { out = i % n_plus_one == 0 ? CudaType{1} : CudaType{0}; }
    int64_t n_plus_one;
};

}  // namespace

void CudaDevice::Identity(const Array& out) {
    CHAINERX_ASSERT(out.ndim() == 2);
    CHAINERX_ASSERT(out.shape()[0] == out.shape()[1]);

    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(IdentityImpl<T>{out.shape()[0]}, out);
    });
}

namespace {

template <typename T>
struct EyeImpl {
    using CudaType = cuda_internal::DataType<T>;
    EyeImpl(int64_t m, int64_t k) : start{k < 0 ? -k * m : k}, stop{m * (m - k)}, step{m + 1} {}
    __device__ void operator()(int64_t i, CudaType& out) {
        out = start <= i && i < stop && (i - start) % step == 0 ? CudaType{1} : CudaType{0};
    }
    int64_t start;
    int64_t stop;
    int64_t step;
};

}  // namespace

void CudaDevice::Eye(int64_t k, const Array& out) {
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [k, &out](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(EyeImpl<T>{out.shape()[1], k}, out);
    });
}

namespace {

template <typename T>
__global__ void SetVecInMat(
        IndexableArray<const T, 1> vec_iarray,
        IndexableArray<T, 2> mat_iarray,
        Indexer<1> vec_indexer,
        Indexer<2> mat_indexer,
        int64_t mat_row_start,
        int64_t mat_col_start) {
    auto mat_it = mat_indexer.It(0);
    for (auto vec_it = vec_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); vec_it; ++vec_it) {
        mat_it.index()[0] = mat_row_start + vec_it.raw_index();
        mat_it.index()[1] = mat_col_start + vec_it.raw_index();
        mat_iarray[mat_it] = vec_iarray[vec_it];
    }
}

}  // namespace

void CudaDevice::Diagflat(const Array& v, int64_t k, const Array& out) {
    CHAINERX_ASSERT(v.ndim() == 1);
    CHAINERX_ASSERT(out.ndim() == 2);

    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // Start indices for the 2-D array axes with applied offset k.
        int64_t row_start{0};
        int64_t col_start{0};

        if (k >= 0) {
            col_start += k;
        } else {
            row_start -= k;
        }

        // Initialize all elements to 0 first instead of conditionally filling in the diagonal.
        Fill(out, T{0});

        IndexableArray<const T, 1> v_iarray{v};
        IndexableArray<T, 2> out_iarray{out};
        Indexer<1> v_indexer{v.shape()};
        Indexer<2> out_indexer{out.shape()};

        // TODO(niboshi): Calculate kMaxBlockSize per device
        std::lock_guard<std::mutex> lock{*cuda_internal::g_mutex};
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&SetVecInMat<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        SetVecInMat<<<grid_size, block_size>>>(v_iarray, out_iarray, v_indexer, out_indexer, row_start, col_start);
    });
}

namespace {

template <typename T>
struct LinspaceImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t i, CudaType& out) {
        double value = n == 1 ? start : (start * (n - 1 - i) + stop * i) / (n - 1);
        out = cuda_numeric_cast<CudaType>(value);
    }
    int64_t n;
    double start;
    double stop;
};

}  // namespace

void CudaDevice::Linspace(double start, double stop, const Array& out) {
    CHAINERX_ASSERT(out.ndim() == 1);
    CHAINERX_ASSERT(out.shape()[0] > 0);

    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        int64_t n = out.shape()[0];
        Elementwise<T>(LinspaceImpl<T>{n, start, stop}, out);
    });
}

}  // namespace cuda
}  // namespace chainerx
