#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/macro.h"
#include "xchainer/reduction_kernel_arg.h"

namespace xchainer {
namespace cuda {
namespace reduce_detail {

static constexpr int kMaxReductionBlockSize{512};
static constexpr int64_t kMaxGridSize{0x7fffffff};

int64_t RoundUpToPowerOf2(int64_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

#define _REDUCE(offset)                               \
    if (tid < offset) {                               \
        impl.Reduce(work[(tid + offset)], work[tid]); \
    }

template <
        typename In,
        typename Out,
        typename ReductionImpl,
        int8_t InNdim = kDynamicNdim,
        int8_t OutNdim = kDynamicNdim,
        int8_t ReduceNdim = kDynamicNdim>
__global__ void ReductionKernel(
        ReductionKernelArg<In, Out, InNdim, OutNdim, ReduceNdim> arg, int out_block_size, int reduce_block_size, ReductionImpl impl) {
    using T = decltype(impl.Identity());

    extern __shared__ __align__(8) uint8_t work_bytes[];
    T* work = reinterpret_cast<T*>(work_bytes);
    int tid = threadIdx.x;

    int64_t reduce_block_offset = tid / out_block_size;

    int64_t out_offset = tid % out_block_size;
    int64_t out_base = blockIdx.x * out_block_size;
    int64_t out_stride = gridDim.x * out_block_size;

    auto it_in = arg.in_indexer.It(0);

    for (auto it_out = arg.out_indexer.It(out_base + out_offset, out_stride); it_out; ++it_out) {
        T accum = impl.Identity();

        // Set output indices in the corresponding indices (out_axis) in input index.
        for (int8_t i_out_dim = 0; i_out_dim < arg.out_indexer.ndim(); ++i_out_dim) {
            it_in.index()[i_out_dim] = it_out.index()[i_out_dim];
        }

        // Linearly compute the partial sum onto reduction blocks.
        for (auto it_reduce = arg.reduce_indexer.It(reduce_block_offset, reduce_block_size); it_reduce; ++it_reduce) {
            // Set reduction indices in the corresponding indices (axis) in input index.
            for (int8_t i_reduce_dim = 0; i_reduce_dim < arg.reduce_indexer.ndim(); ++i_reduce_dim) {
                it_in.index()[arg.out_indexer.ndim() + i_reduce_dim] = it_reduce.index()[i_reduce_dim];
            }

            int64_t i_reduce = it_reduce.raw_index();
            impl.Reduce(impl.MapIn(arg.in[it_in], i_reduce), accum);
        }

        if (out_block_size < 512) {
            work[tid] = accum;
            __syncthreads();
            if (out_block_size <= 256) {
                _REDUCE(256);
                __syncthreads();
                if (out_block_size <= 128) {
                    _REDUCE(128);
                    __syncthreads();
                    if (out_block_size <= 64) {
                        _REDUCE(64);
                        __syncthreads();
                        if (out_block_size <= 32) {
                            _REDUCE(32);
                            __syncthreads();
                            if (out_block_size <= 16) {
                                _REDUCE(16);
                                __syncthreads();
                                if (out_block_size <= 8) {
                                    _REDUCE(8);
                                    __syncthreads();
                                    if (out_block_size <= 4) {
                                        _REDUCE(4);
                                        __syncthreads();
                                        if (out_block_size <= 2) {
                                            _REDUCE(2);
                                            __syncthreads();
                                            if (out_block_size <= 1) {
                                                _REDUCE(1);
                                                __syncthreads();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            accum = work[tid];
            __syncthreads();
        }
        if (reduce_block_offset == 0 && it_out) {
            arg.out[it_out] = impl.MapOut(accum);
        }
    }
}

#undef _REDUCE

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
    ReductionArg arg{in, axis, out};

    static const int64_t kMaxBlockSize = std::min(
            reduce_detail::kMaxReductionBlockSize,
            CudaOccupancyMaxPotentialBlockSize(&reduce_detail::ReductionKernel<In, Out, ReductionImpl>).block_size);

    int64_t reduce_total_size_pow2 = reduce_detail::RoundUpToPowerOf2(std::max(int64_t{1}, arg.reduce_shape().GetTotalSize()));

    int64_t reduce_block_size = std::min(kMaxBlockSize, reduce_total_size_pow2);
    int64_t out_block_size = kMaxBlockSize / reduce_block_size;
    int64_t out_block_num = (arg.out_shape().GetTotalSize() + out_block_size - 1) / out_block_size;

    int64_t block_size = kMaxBlockSize;
    int64_t grid_size = std::min(reduce_detail::kMaxGridSize, out_block_num);
    int64_t shared_mem_size = sizeof(decltype(impl.Identity())) * block_size;

    assert(arg.in_shape().ndim() == arg.out_shape().ndim() + arg.reduce_shape().ndim());
#ifdef NDEBUG  // Optimize only in Release build to save time on development
    // TODO(sonots): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    switch (arg.in_strides().ndim()) {
        case 1:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 1, 0, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 1, 1, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
        case 2:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 0, 2>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 1, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 2:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 2, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
        case 3:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 0, 3>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 1, 2>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 2:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 2, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 3:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 3, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
        case 4:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 0, 4>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 1, 3>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 2:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 2, 2>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 3:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 3, 1>(arg), out_block_size, reduce_block_size, impl);
                    return;
                case 4:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 4, 0>(arg), out_block_size, reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
    }
#endif

    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
            MakeReductionKernelArg<In, Out>(arg), out_block_size, reduce_block_size, impl);
}

}  // namespace cuda
}  // namespace xchainer
