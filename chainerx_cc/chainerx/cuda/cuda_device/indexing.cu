#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <type_traits>

#include <gsl/gsl>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/kernels/indexing.h"
#include "chainerx/macro.h"
#include "chainerx/routines/indexing.h"
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

template <typename T, typename TIndex, int8_t kNdim>
__global__ void TakeCudaKernel(
        IndexableArray<const T, kNdim> a,
        IndexableArray<const TIndex, kNdim> indices,
        IndexableArray<T, kNdim> out,
        Indexer<kNdim> a_indexer,
        Indexer<kNdim> indices_indexer,
        Indexer<kNdim> out_indexer,
        TIndex num_indices,
        TIndex target_dim,
        TIndex right_dim,
        TIndex num_iters,
        IndexBoundsMode mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_iters) return;
    TIndex left_idx = idx / (num_indices * right_dim);
    TIndex index_idx = idx % (num_indices * right_dim);
    TIndex right_idx = index_idx % right_dim;
    TIndex index_pos = index_idx / right_dim;
    TIndex index = indices[indices_indexer.It(index_pos)];
    if (mode == IndexBoundsMode::kWrap || mode == IndexBoundsMode::kDefault) {
        if (index < 0) {
            index = target_dim - ((-index + target_dim - 1) % target_dim + 1);
        } else {
            index = index % target_dim;
        }
    } else if (mode == IndexBoundsMode::kClip) {
        index = max(TIndex{0}, min(index, TIndex{target_dim} - 1));
    }
    TIndex yi = (left_idx * num_indices + index_pos) * right_dim + right_idx;
    TIndex xi = (left_idx * target_dim + index) * right_dim + right_idx;
    out[out_indexer.It(yi)] = a[a_indexer.It(xi)];
}

template <typename T, typename TIndex>
__global__ void AddAtWrapCudaKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<const T> b_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const TIndex> indices_iarray,
        Indexer<> b_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        TIndex common_total_size,
        TIndex axis_dim) {
    static_assert(std::is_same<TIndex, int64_t>::value || std::is_same<TIndex, int32_t>::value, "");
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        TIndex axis_pos = static_cast<TIndex>(it.raw_index()) / common_total_size;
        TIndex common_pos = static_cast<TIndex>(it.raw_index()) % common_total_size;

        cuda_internal::DataType<T> out_value = cuda_internal::StorageToDataType<const T>(a_iarray[it]);

        for (auto it_indices = indices_indexer.It(0); it_indices; ++it_indices) {
            TIndex index = indices_iarray[it_indices];

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

template <typename T, typename TIndex>
__global__ void AddAtClipCudaKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<const T> b_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const TIndex> indices_iarray,
        Indexer<> b_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        TIndex common_total_size,
        TIndex axis_dim) {
    static_assert(std::is_same<TIndex, int64_t>::value || std::is_same<TIndex, int32_t>::value, "");
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        TIndex axis_pos = static_cast<TIndex>(it.raw_index()) / common_total_size;
        TIndex common_pos = static_cast<TIndex>(it.raw_index()) % common_total_size;

        cuda_internal::DataType<T> out_value = cuda_internal::StorageToDataType<const T>(a_iarray[it]);

        for (auto it_indices = indices_indexer.It(0); it_indices; ++it_indices) {
            TIndex index = indices_iarray[it_indices];

            index = max(TIndex{0}, min(index, axis_dim - 1));
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

Array CollapseNdArray(const Array& a) {
    Strides new_strides;
    Shape new_shape;
    const Strides& strides = a.strides();
    const Shape& shape = a.shape();

    size_t size = strides.size();

    for (size_t i = 0; i < size; i++) {
        if (!new_shape.empty() && new_strides.back() == strides[i] * shape[i]) {
            new_shape.back() *= shape[i];
            new_strides.back() = strides[i];
        } else {
            new_shape.push_back(shape[i]);
            new_strides.push_back(strides[i]);
        }
    }

    return internal::MakeArray(new_shape, new_strides, a.dtype(), a.device(), a.data(), a.offset());
}

template <typename TIndex>
void TakeImpl(Device& device, const Array& a, const Array& indices, int8_t axis, const Array& out, IndexBoundsMode mode) {
    if (mode == IndexBoundsMode::kRaise) {
        throw BackendError{"Take with mode='raise' is not supported with CUDA backend"};
    }
    static_assert(std::is_same<TIndex, int64_t>::value || std::is_same<TIndex, int32_t>::value, "");
    CHAINERX_ASSERT(
            (std::is_same<TIndex, int64_t>::value && indices.dtype() == Dtype::kInt64) ||
            (std::is_same<TIndex, int32_t>::value && indices.dtype() == Dtype::kInt32));
    device.CheckDevicesCompatible(a, indices, out);

    CudaSetDeviceScope scope{device.index()};

    VisitDtype(out.dtype(), [&a, &indices, axis, &out, mode](auto pt) {
        using T = typename decltype(pt)::type;
        TIndex num_indices = indices.GetTotalSize();
        TIndex target_dim = a.shape()[axis];
        TIndex num_iters = out.GetTotalSize();
        TIndex right_dim = std::accumulate(a.shape().begin() + axis + 1, a.shape().end(), int64_t{1}, std::multiplies<>());
        static const int k1DMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&TakeCudaKernel<T, TIndex, 1>).block_size;
        static const int kNDMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&TakeCudaKernel<T, TIndex, kDynamicNdim>).block_size;
        Array af = CollapseNdArray(a);
        Array indicesf = CollapseNdArray(indices);
        Array outf = CollapseNdArray(out);
        // To get the maximum performance we need to explicitly instantiate the 1D
        // version of the indexable array. If only one of the 3 arrays has more than one dim
        // the performance is greatly affected. In such case we just treat all the arrays equally
        // to simplify the code logic
        if (af.ndim() == 1 && indicesf.ndim() == 1 && outf.ndim() == 1) {
            IndexableArray<const T, 1> a_iarray{af};
            IndexableArray<const TIndex, 1> indices_iarray{indicesf};
            IndexableArray<T, 1> out_iarray{outf};
            Indexer<1> a_indexer{af.shape()};
            Indexer<1> indices_indexer{indicesf.shape()};
            Indexer<1> out_indexer{outf.shape()};

            TakeCudaKernel<T, TIndex, 1><<<(num_iters + k1DMaxBlockSize - 1) / k1DMaxBlockSize, k1DMaxBlockSize>>>(
                    a_iarray,
                    indices_iarray,
                    out_iarray,
                    a_indexer,
                    indices_indexer,
                    out_indexer,
                    num_indices,
                    target_dim,
                    right_dim,
                    num_iters,
                    mode);
        } else {
            IndexableArray<const T> a_iarray{af};
            IndexableArray<const TIndex> indices_iarray{indicesf};
            IndexableArray<T> out_iarray{outf};
            Indexer<> a_indexer{af.shape()};
            Indexer<> indices_indexer{indicesf.shape()};
            Indexer<> out_indexer{outf.shape()};

            TakeCudaKernel<T, TIndex, kDynamicNdim><<<(num_iters + kNDMaxBlockSize - 1) / kNDMaxBlockSize, kNDMaxBlockSize>>>(
                    a_iarray,
                    indices_iarray,
                    out_iarray,
                    a_indexer,
                    indices_indexer,
                    out_indexer,
                    num_indices,
                    target_dim,
                    right_dim,
                    num_iters,
                    mode);
        }
        return;
    });
}

template <typename TIndex>
void AddAtImpl(Device& device, const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out, IndexBoundsMode mode) {
    // TODO(niboshi): Current implementation only distributes output elements in respective threads. Summation on the indices is performed
    // serially in each thread. This implementation can be improved by distributing indices as well, possibly using atomicAdd.

    static_assert(std::is_same<TIndex, int64_t>::value || std::is_same<TIndex, int32_t>::value, "");
    CHAINERX_ASSERT(
            (std::is_same<TIndex, int64_t>::value && indices.dtype() == Dtype::kInt64) ||
            (std::is_same<TIndex, int32_t>::value && indices.dtype() == Dtype::kInt32));
    CHAINERX_ASSERT(a.shape() == out.shape());
    device.CheckDevicesCompatible(a, indices, out);

    CudaSetDeviceScope scope{device.index()};

    VisitDtype(out.dtype(), [&a, &indices, axis, &b, &out, mode](auto pt) {
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

        IndexableArray<const TIndex> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        TIndex common_total_size = gsl::narrow<TIndex>(a_indexer.total_size() / a_shape[0]);

        TIndex axis_dim = gsl::narrow<TIndex>(a_shape[0]);

        int64_t total_size = out_indexer.total_size();
        static const int kWrapMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddAtWrapCudaKernel<T, TIndex>).block_size;
        static const int kClipMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddAtClipCudaKernel<T, TIndex>).block_size;
        int64_t grid_size;
        int64_t block_size;
        switch (mode) {
            case IndexBoundsMode::kRaise:
                throw BackendError{"Take with mode='raise' is not supported with CUDA backend"};
            case IndexBoundsMode::kDefault:
            case IndexBoundsMode::kWrap:
                grid_size = (total_size + kWrapMaxBlockSize - 1) / kWrapMaxBlockSize;
                block_size = std::min<int64_t>(total_size, kClipMaxBlockSize);
                AddAtWrapCudaKernel<<<grid_size, block_size>>>(
                        a_iarray,
                        b_iarray,
                        out_iarray,
                        indices_iarray,
                        b_indexer,
                        out_indexer,
                        indices_indexer,
                        common_total_size,
                        axis_dim);
                break;
            case IndexBoundsMode::kClip:
                grid_size = (total_size + kClipMaxBlockSize - 1) / kClipMaxBlockSize;
                block_size = std::min<int64_t>(total_size, kClipMaxBlockSize);
                AddAtClipCudaKernel<<<grid_size, block_size>>>(
                        a_iarray,
                        b_iarray,
                        out_iarray,
                        indices_iarray,
                        b_indexer,
                        out_indexer,
                        indices_indexer,
                        common_total_size,
                        axis_dim);
                break;
        }
    });
}

class CudaTakeKernel : public TakeKernel {
public:
    void Call(const Array& a, const Array& indices, int8_t axis, const Array& out, IndexBoundsMode mode) override {
        Device& device = a.device();
        CHAINERX_ASSERT(GetKind(indices.dtype()) == DtypeKind::kInt || GetKind(indices.dtype()) == DtypeKind::kUInt);
        device.CheckDevicesCompatible(a, indices, out);

        CudaSetDeviceScope scope{device.index()};

        if (indices.dtype() == Dtype::kInt64) {
            TakeImpl<int64_t>(device, a, indices, axis, out, mode);
        } else {
            const Array& indices_cast = indices.dtype() == Dtype::kInt32 ? indices : indices.AsType(Dtype::kInt32);
            TakeImpl<int32_t>(device, a, indices_cast, axis, out, mode);
        }
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(TakeKernel, CudaTakeKernel);

class CudaAddAtKernel : public AddAtKernel {
public:
    void Call(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out, IndexBoundsMode mode) override {
        Device& device = a.device();
        CHAINERX_ASSERT(GetKind(indices.dtype()) == DtypeKind::kInt || GetKind(indices.dtype()) == DtypeKind::kUInt);
        device.CheckDevicesCompatible(a, indices, out);

        CudaSetDeviceScope scope{device.index()};

        if (indices.dtype() == Dtype::kInt64) {
            AddAtImpl<int64_t>(device, a, indices, axis, b, out, mode);
        } else {
            const Array& indices_cast = indices.dtype() == Dtype::kInt32 ? indices : indices.AsType(Dtype::kInt32);
            AddAtImpl<int32_t>(device, a, indices_cast, axis, b, out, mode);
        }
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(AddAtKernel, CudaAddAtKernel);

template <typename T>
struct WhereImpl {
    using CudaType = cuda_internal::DataType<T>;
    __device__ void operator()(int64_t /*i*/, bool condition, CudaType x, CudaType y, CudaType& out) { out = condition ? x : y; }
};

class CudaWhereKernel : public WhereKernel {
public:
    void Call(const Array& condition, const Array& x, const Array& y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, x, y, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        Dtype out_dtype = out.dtype();
        const Array& x_cast = x.dtype() != out_dtype ? x.AsType(out_dtype) : x;
        const Array& y_cast = y.dtype() != out_dtype ? y.AsType(out_dtype) : y;

        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto x_pt) {
            using T = typename decltype(x_pt)::type;
            Elementwise<const bool, const T, const T, T>(WhereImpl<T>{}, condition_cast, x_cast, y_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(WhereKernel, CudaWhereKernel);

template <typename T>
struct WhereAASImpl {
    using CudaType = cuda_internal::DataType<T>;
    CudaType y;
    __device__ void operator()(int64_t /*i*/, bool condition, CudaType x, CudaType& out) { out = condition ? x : y; }
};

class CudaWhereAASKernel : public WhereAASKernel {
public:
    void Call(const Array& condition, const Array& x, Scalar y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, x, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        Dtype out_dtype = out.dtype();
        const Array& x_cast = x.dtype() != out_dtype ? x.AsType(out_dtype) : x;

        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto x_pt) {
            using T = typename decltype(x_pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const bool, const T, T>(WhereAASImpl<T>{static_cast<CudaType>(y)}, condition_cast, x_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(WhereAASKernel, CudaWhereAASKernel);

template <typename T>
struct WhereASAImpl {
    using CudaType = cuda_internal::DataType<T>;
    CudaType x;
    __device__ void operator()(int64_t /*i*/, bool condition, CudaType y, CudaType& out) { out = condition ? x : y; }
};

class CudaWhereASAKernel : public WhereASAKernel {
public:
    void Call(const Array& condition, Scalar x, const Array& y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, y, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        Dtype out_dtype = out.dtype();
        const Array& y_cast = y.dtype() != out_dtype ? y.AsType(out_dtype) : y;

        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto x_pt) {
            using T = typename decltype(x_pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const bool, const T, T>(WhereASAImpl<T>{static_cast<CudaType>(x)}, condition_cast, y_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(WhereASAKernel, CudaWhereASAKernel);

template <typename T>
struct WhereASSImpl {
    using CudaType = cuda_internal::DataType<T>;
    CudaType x;
    CudaType y;
    __device__ void operator()(int64_t /*i*/, bool condition, CudaType& out) { out = condition ? x : y; }
};

class CudaWhereASSKernel : public WhereASSKernel {
public:
    void Call(const Array& condition, Scalar x, Scalar y, const Array& out) override {
        Device& device = condition.device();
        device.CheckDevicesCompatible(condition, out);
        const Array& condition_cast = condition.dtype() != Dtype::kBool ? condition.AsType(Dtype::kBool) : condition;

        CudaSetDeviceScope scope{device.index()};
        VisitDtype(out.dtype(), [&](auto x_pt) {
            using T = typename decltype(x_pt)::type;
            using CudaType = cuda_internal::DataType<T>;
            Elementwise<const bool, T>(WhereASSImpl<T>{static_cast<CudaType>(x), static_cast<CudaType>(y)}, condition_cast, out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(WhereASSKernel, CudaWhereASSKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
