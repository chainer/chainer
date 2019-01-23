#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <numeric>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace cuda {

namespace {

// Makes axes for permutation that moves [first_axis, last_axis) to the head.
Axes MakeRollingPermutation(int8_t first_axis, int8_t last_axis, int8_t ndim) {
    CHAINERX_ASSERT(0 <= first_axis);
    CHAINERX_ASSERT(first_axis < last_axis);
    CHAINERX_ASSERT(last_axis <= ndim);

    Axes permutation{};
    permutation.resize(ndim);
    auto head_end = permutation.begin() + (last_axis - first_axis);
    auto last = permutation.begin() + last_axis;
    std::iota(permutation.begin(), head_end, first_axis);
    std::iota(head_end, last, int8_t{0});
    std::iota(last, permutation.end(), last_axis);
    return permutation;
}

template <typename T>
__global__ void TakeKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const int64_t> indices_iarray,
        Indexer<> a_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        int64_t common_total_size,
        int64_t axis_dim) {
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        int64_t indices_pos = it.raw_index() / common_total_size;
        int64_t common_pos = it.raw_index() % common_total_size;

        int64_t index = indices_iarray[indices_indexer.It(indices_pos)];
        if (index < 0) {
            index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
        } else {
            index = index % axis_dim;
        }
        CHAINERX_ASSERT(0 <= index);
        CHAINERX_ASSERT(index < axis_dim);

        out_iarray[it] = a_iarray[a_indexer.It(index * common_total_size + common_pos)];
    }
}

template <typename T>
__global__ void AddAtKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<const T> b_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const int64_t> indices_iarray,
        Indexer<> b_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        int64_t common_total_size,
        int64_t axis_dim) {
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        int64_t axis_pos = it.raw_index() / common_total_size;
        int64_t common_pos = it.raw_index() % common_total_size;

        cuda_internal::DataType<T> out_value = cuda_internal::StorageToDataType<const T>(a_iarray[it]);

        for (auto it_indices = indices_indexer.It(0); it_indices; ++it_indices) {
            int64_t index = indices_iarray[it_indices];

            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            CHAINERX_ASSERT(0 <= index);
            CHAINERX_ASSERT(index < axis_dim);

            if (index == axis_pos) {
                out_value += cuda_internal::StorageToDataType<const T>(
                        b_iarray[b_indexer.It(it_indices.raw_index() * common_total_size + common_pos)]);
            }
        }

        out_iarray[it] = cuda_internal::DataToStorageType<T>(out_value);
    }
}

}  // namespace

void CudaDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CheckDevicesCompatible(a, indices, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // a and out are transposed as follows.
        // a:       (Ni..., N,     Nj...) => (N,     Ni..., Nj...)
        // out:     (Ni..., Nk..., Nj...) => (Nk..., Ni..., Nj...)
        //
        // indices is used as is.
        // indices: (Nk...)

        IndexableArray<const T> a_iarray{a};
        Axes a_perm = MakeRollingPermutation(axis, axis + 1, a.ndim());
        a_iarray.Permute(a_perm);
        Shape a_shape = internal::TransposeShape(a.shape(), a_perm);
        Indexer<> a_indexer{a_shape};

        IndexableArray<T> out_iarray{out};
        Axes out_perm = MakeRollingPermutation(axis, axis + indices.ndim(), out.ndim());
        out_iarray.Permute(out_perm);
        Shape out_shape = internal::TransposeShape(out.shape(), out_perm);
        Indexer<> out_indexer{out_shape};

        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        int64_t common_total_size = a_indexer.total_size() / a_shape[0];

        // TODO(niboshi): Calculate kMaxBlockSize per device
        std::lock_guard<std::mutex> lock{*cuda_internal::g_mutex};
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&TakeKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        TakeKernel<<<grid_size, block_size>>>(
                a_iarray, out_iarray, indices_iarray, a_indexer, out_indexer, indices_indexer, common_total_size, a_shape[0]);
    });
}

void CudaDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    // TODO(niboshi): Current implementation only distributes output elements in respective threads. Summation on the indices is performed
    // serially in each thread. This implementation can be improved by distributing indices as well, possibly using atomicAdd.

    CHAINERX_ASSERT(a.shape() == out.shape());
    CheckDevicesCompatible(a, indices, out);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // b and out are transposed as follows.
        // a:       (Ni..., N,     Nj...) => (N,     Ni..., Nj...)
        // b:       (Ni..., Nk..., Nj...) => (Nk..., Ni..., Nj...)
        // out:     (Ni..., N    , Nj...) => (N    , Ni..., Nj...)
        //
        // indices is used as is.
        // indices: (Nk...)

        IndexableArray<const T> a_iarray{a};
        Axes a_perm = MakeRollingPermutation(axis, axis + 1, a.ndim());
        a_iarray.Permute(a_perm);
        Shape a_shape = internal::TransposeShape(a.shape(), a_perm);
        Indexer<> a_indexer{a_shape};

        IndexableArray<const T> b_iarray{b};
        Axes b_perm = MakeRollingPermutation(axis, axis + indices.ndim(), b.ndim());
        b_iarray.Permute(b_perm);
        Shape b_shape = internal::TransposeShape(b.shape(), b_perm);
        Indexer<> b_indexer{b_shape};

        IndexableArray<T> out_iarray{out};
        Axes out_perm = MakeRollingPermutation(axis, axis + 1, out.ndim());
        out_iarray.Permute(out_perm);
        Shape out_shape = internal::TransposeShape(out.shape(), out_perm);
        Indexer<> out_indexer{out_shape};

        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        int64_t common_total_size = a_indexer.total_size() / a_shape[0];

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddAtKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        AddAtKernel<<<grid_size, block_size>>>(
                a_iarray, b_iarray, out_iarray, indices_iarray, b_indexer, out_indexer, indices_indexer, common_total_size, a_shape[0]);
    });
}

}  // namespace cuda
}  // namespace chainerx
