#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <type_traits>

#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/macro.h"
#include "chainerx/reduction_kernel_arg.h"

namespace chainerx {
namespace cuda {
namespace reduce_detail {

static constexpr int kMaxReductionBlockSize{512};
static constexpr int64_t kMaxGridSize{0x7fffffff};

inline int64_t RoundUpToPowerOf2(int64_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

template <typename In, typename Out, typename ReductionImpl, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim>
__global__ void ReductionKernel(
        ReductionKernelArg<In, Out, InNdim, OutNdim> arg, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    using AccumType = decltype(impl.Identity());

    extern __shared__ __align__(8) uint8_t work_bytes[];
    AccumType* work = reinterpret_cast<AccumType*>(work_bytes);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    int tid = threadIdx.x;

    int64_t reduce_block_offset = tid / out_block_size;
    int64_t reduce_offset = reduce_block_offset * arg.out_indexer.total_size();
    int64_t reduce_stride = reduce_block_size * arg.out_indexer.total_size();

    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    auto it_in = arg.in_indexer.It(0, reduce_stride);

    for (auto it_out = arg.out_indexer.It(out_base + out_offset, out_stride); it_out; ++it_out) {
        AccumType accum = impl.Identity();

        int64_t i_reduce = reduce_block_offset;
        for (it_in.Restart(it_out.raw_index() + reduce_offset); it_in; ++it_in, i_reduce += reduce_block_size) {
            impl.Reduce(impl.MapIn(cuda_internal::StorageToDataType<const In>(arg.in[it_in]), i_reduce), accum);
        }

        if (out_block_size <= kMaxReductionBlockSize / 2) {
            work[tid] = accum;
            __syncthreads();
            // NOTE: Compiler optimizes to unroll this loop
            for (int stride = kMaxReductionBlockSize / 2; stride > 0; stride >>= 1) {
                if (out_block_size <= stride) {
                    if (tid < stride) {
                        impl.Reduce(work[tid + stride], work[tid]);
                    }
                    __syncthreads();
                }
            }
            accum = work[tid];
            __syncthreads();
        }
        if (reduce_block_offset == 0 && it_out) {
            arg.out[it_out] = cuda_internal::DataToStorageType<Out>(impl.MapOut(accum));
        }
    }
}

template <typename In, typename Out, typename ReductionImpl, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim>
__global__ void ScanKernel(
        ReductionKernelArg<In, Out, InNdim, OutNdim> arg,
        int out_block_size,
        int reduce_block_size,
        ReductionImpl impl,
        int64_t reduce_len) {
    int tid = threadIdx.x;

    int64_t len = arg.out_indexer.total_size() / reduce_len;
    int64_t reduce_block_offset = tid / out_block_size;
    int64_t reduce_offset = reduce_block_offset * len;
    int64_t reduce_stride = reduce_block_size * len;

    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    auto reduce = [&impl, &arg](auto& it_from, auto& it_to) {
        auto from = cuda_internal::StorageToDataType<Out>(arg.out[it_from]);
        auto& to = cuda_internal::StorageToDataType<Out>(arg.out[it_to]);
        impl.Reduce(from, to);
        ++it_from;
        ++it_to;
    };

    for (int64_t i = out_base + out_offset; i < len; i += out_stride) {
        // Copy input array to output array
        auto it_in = arg.in_indexer.It(i + reduce_offset, reduce_stride);
        auto it_out = arg.out_indexer.It(i + reduce_offset, reduce_stride);
        for (int64_t j = reduce_block_offset; j < reduce_len; j += reduce_block_size, ++it_in, ++it_out) {
            auto value = cuda_internal::StorageToDataType<const In>(arg.in[it_in]);
            arg.out[it_out] = cuda_internal::DataToStorageType<Out>(impl.MapIn(value, j));
        }
        __syncthreads();

        int64_t stride = 1;

        // Up-Sweep Phase
        for (stride = 1; stride * 2 <= reduce_len; stride <<= 1) {
            int64_t index_from = reduce_block_offset * stride * 2 + stride - 1;
            int64_t index_to = index_from + stride;
            auto it_from = arg.out_indexer.It(i + index_from * len, reduce_stride * stride * 2);
            auto it_to = arg.out_indexer.It(i + index_to * len, reduce_stride * stride * 2);
            for (int64_t j = index_to; j < reduce_len; j += reduce_block_size * stride * 2) {
                reduce(it_from, it_to);
            }
            __syncthreads();
        }

        // Down-Sweep Phase
        for (; stride >= 1; stride >>= 1) {
            int64_t index_from = reduce_block_offset * stride * 2 + stride * 2 - 1;
            int64_t index_to = index_from + stride;
            auto it_from = arg.out_indexer.It(i + index_from * len, reduce_stride * stride * 2);
            auto it_to = arg.out_indexer.It(i + index_to * len, reduce_stride * stride * 2);
            for (int64_t j = index_to; j < reduce_len; j += reduce_block_size * stride * 2) {
                reduce(it_from, it_to);
            }
            __syncthreads();
        }
    }
}

}  // namespace reduce_detail

// Computes the reduction of the input and stores into the output array.
//
// `ReductionImpl` is required to provide the following device member function.
// T can be arbitrary but should be common between these functions.
//
// - T Identity();
//       Returns the initial value of reduction.
// - T MapIn(In in, int64_t index);
//       Applies pre-reduction mapping of the input and its index.
// - void Reduce(T next, T& accum);
//       Accumulates the iterated value to accum.
// - Out MapOut(T accum);
//       Applies post-reduction mapping of the output.
//
// Example:
//     Simple summation over a float array can be implemented as the following reduction impl.
//
//         struct SumImpl {
//             __device__ float Identity() { return 0; }
//             __device__ float MapIn(float in, int64_t /*index*/) { return in; }
//             __device__ void Reduce(float next, float& accum) { accum += next; }
//             __device__ float MapOut(float accum) { return accum; }
//         };
//
//     Then, it can be passed to Reduce like: Reduce(input, axis, output, SumImpl{});
template <typename In, typename Out, typename ReductionImpl>
void Reduce(const Array& in, const Axes& axis, const Array& out, ReductionImpl&& impl) {
    if (out.GetTotalSize() == 0) {
        return;
    }

    ReductionArg arg{in, axis, out};

    // TODO(niboshi): Calculate kMaxBlockSize per device
    std::lock_guard<std::mutex> lock{*cuda_internal::g_mutex};
    static const int64_t kMaxBlockSize = std::min(
            reduce_detail::kMaxReductionBlockSize,
            CudaOccupancyMaxPotentialBlockSize(&reduce_detail::ReductionKernel<In, Out, ReductionImpl>).block_size);

    int64_t reduce_total_size_pow2 =
            reduce_detail::RoundUpToPowerOf2(std::max(int64_t{1}, arg.in_shape().GetTotalSize() / arg.out_shape().GetTotalSize()));

    int64_t reduce_block_size = std::min(kMaxBlockSize, reduce_total_size_pow2);
    int64_t out_block_size = kMaxBlockSize / reduce_block_size;
    int64_t out_block_num = (arg.out_shape().GetTotalSize() + out_block_size - 1) / out_block_size;

    int64_t block_size = kMaxBlockSize;
    int64_t grid_size = std::min(reduce_detail::kMaxGridSize, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity())) * block_size;

#ifdef NDEBUG  // Optimize only in Release build to save time on development
    // TODO(sonots): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    // Currently, only contiguous output arrays are optimized.
    switch (arg.in_strides().ndim()) {
        case 1:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 1, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 1, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            break;
        case 2:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            break;
        case 3:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            break;
        case 4:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            break;
    }
#endif  // NDEBUG

    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
            MakeReductionKernelArg<In, Out>(arg), out_block_size, reduce_block_size, impl);
}

template <typename In, typename Out, typename ReductionImpl>
void Scan(const Array& in, int8_t axis, const Array& out, ReductionImpl&& impl) {
    if (out.GetTotalSize() == 0) {
        return;
    }

    ReductionArg arg{in, Axes{axis}, out};
    int64_t reduce_len = in.shape()[axis];

    // TODO(niboshi): Calculate kMaxBlockSize per device
    std::lock_guard<std::mutex> lock{*cuda_internal::g_mutex};
    static const int64_t kMaxBlockSize = std::min(
            reduce_detail::kMaxReductionBlockSize,
            CudaOccupancyMaxPotentialBlockSize(&reduce_detail::ReductionKernel<In, Out, ReductionImpl>).block_size);

    int64_t reduce_total_size_pow2 = reduce_detail::RoundUpToPowerOf2(std::max(int64_t{1}, reduce_len));

    int64_t reduce_block_size = std::min(kMaxBlockSize, reduce_total_size_pow2);
    int64_t out_block_size = kMaxBlockSize / reduce_block_size;
    int64_t out_block_num = (arg.in_shape().GetTotalSize() / reduce_len + out_block_size - 1) / out_block_size;

    int64_t block_size = kMaxBlockSize;
    int64_t grid_size = std::min(reduce_detail::kMaxGridSize, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity())) * block_size;

#ifdef NDEBUG  // Optimize only in Release build to save time on development
    // TODO(sonots): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    // Currently, only contiguous output arrays are optimized.
    if (arg.in_strides().ndim() == 1 && arg.out_strides().ndim() == 1) {
        reduce_detail::ScanKernel<<<grid_size, block_size, shared_mem_size>>>(
                MakeReductionKernelArg<In, Out, 1, 1>(arg), out_block_size, reduce_block_size, impl, reduce_len);
        return;
    }
#endif  // NDEBUG

    reduce_detail::ScanKernel<<<grid_size, block_size, shared_mem_size>>>(
            MakeReductionKernelArg<In, Out>(arg), out_block_size, reduce_block_size, impl, reduce_len);
}

}  // namespace cuda
}  // namespace chainerx
