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

static constexpr int kMaxReductionBlockSize = 512;

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

template <
        typename In,
        typename Out,
        typename ReductionImpl,
        int8_t InNdim = kDynamicNdim,
        int8_t OutNdim = kDynamicNdim,
        int8_t ReduceNdim = kDynamicNdim>
__global__ void ReductionKernel(ReductionKernelArg<In, Out, InNdim, OutNdim, ReduceNdim> arg, int reduce_block_size, ReductionImpl impl) {
    using T = decltype(impl.Identity());

    extern __shared__ __align__(8) uint8_t work_bytes[];
    T* work = reinterpret_cast<T*>(work_bytes);
    int tid = threadIdx.x;
    int reduce_blocks_per_grid = (blockDim.x + reduce_block_size - 1) / reduce_block_size * gridDim.x;

    auto it_in = arg.in_indexer.It(0);

    for (auto it_out = arg.out_indexer.It(blockIdx.x, gridDim.x * reduce_blocks_per_grid); it_out; ++it_out) {
        T accum = impl.Identity();

        // Set output indices in the corresponding indices (out_axis) in src_index.
        for (int8_t i_out_dim = 0; i_out_dim < arg.out_indexer.ndim(); ++i_out_dim) {
            it_in.index()[i_out_dim] = it_out.index()[i_out_dim];
        }

        // Linearly compute the partial sum into at most kMaxReductionBlockSize values.
        for (auto it_reduce = arg.reduce_indexer.It(tid, reduce_block_size); it_reduce; ++it_reduce) {
            // Set reduction indices in the corresponding indices (axis) in src_index.
            for (int8_t i_reduce_dim = 0; i_reduce_dim < arg.reduce_indexer.ndim(); ++i_reduce_dim) {
                it_in.index()[arg.out_indexer.ndim() + i_reduce_dim] = it_reduce.index()[i_reduce_dim];
            }

            impl.Reduce(impl.MapIn(arg.in[it_in], it_reduce.raw_index()), accum);
        }

        if (reduce_block_size >= 2) {
            // Synchronize partial sums
            work[tid] = accum;
            __syncthreads();

            // Reduction
            if (reduce_block_size > 2) {
                if (reduce_block_size > 4) {
                    if (reduce_block_size > 8) {
                        if (reduce_block_size > 16) {
                            if (reduce_block_size > 32) {
                                if (reduce_block_size > 64) {
                                    if (reduce_block_size > 128) {
                                        if (reduce_block_size > 256) {
                                            static_assert(kMaxReductionBlockSize == 512, "");

                                            if (tid < 256) {
                                                impl.Reduce(work[tid + 256], work[tid]);
                                            }
                                            __syncthreads();
                                        }
                                        if (tid < 128) {
                                            impl.Reduce(work[tid + 128], work[tid]);
                                        }
                                        __syncthreads();
                                    }
                                    if (tid < 64) {
                                        impl.Reduce(work[tid + 64], work[tid]);
                                    }
                                    __syncthreads();
                                }
                                if (tid < 32) {
                                    impl.Reduce(work[tid + 32], work[tid]);
                                }
                                __syncthreads();
                            }
                            if (tid < 16) {
                                impl.Reduce(work[tid + 16], work[tid]);
                            }
                            __syncthreads();
                        }
                        if (tid < 8) {
                            impl.Reduce(work[tid + 8], work[tid]);
                        }
                        __syncthreads();
                    }
                    if (tid < 4) {
                        impl.Reduce(work[tid + 4], work[tid]);
                    }
                    __syncthreads();
                }
                if (tid < 2) {
                    impl.Reduce(work[tid + 2], work[tid]);
                }
                __syncthreads();
            }
            accum = work[0];
            impl.Reduce(work[1], accum);
        }
        // Store the output value
        if (tid == 0) {
            arg.out[it_out] = impl.MapOut(accum);
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
    ReductionArg arg{in, axis, out};

    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&reduce_detail::ReductionKernel<In, Out, ReductionImpl>).block_size;

    int reduce_block_size = static_cast<int>(std::min(
            static_cast<int64_t>(reduce_detail::kMaxReductionBlockSize),
            reduce_detail::RoundUpToPowerOf2(std::max(int64_t{1}, arg.reduce_shape().GetTotalSize()))));
    int block_size = std::min(kMaxBlockSize, reduce_block_size);
    int64_t total_reduce_blocks = arg.out_shape().GetTotalSize();
    int64_t grid_size = total_reduce_blocks;
    size_t shared_mem_size = sizeof(decltype(impl.Identity())) * reduce_block_size;

    assert(arg.in_shape().ndim() == arg.out_shape().ndim() + arg.reduce_shape().ndim());
#ifdef NDEBUG  // Optimize only in Release build to save time on development
    // TODO(sonots): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    switch (arg.in_strides().ndim()) {
        case 1:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 1, 0, 1>(arg), reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 1, 1, 0>(arg), reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
        case 2:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 0, 2>(arg), reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 1, 1>(arg), reduce_block_size, impl);
                    return;
                case 2:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 2, 2, 0>(arg), reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
        case 3:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 0, 3>(arg), reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 1, 2>(arg), reduce_block_size, impl);
                    return;
                case 2:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 2, 1>(arg), reduce_block_size, impl);
                    return;
                case 3:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 3, 3, 0>(arg), reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
        case 4:
            switch (arg.out_strides().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 0, 4>(arg), reduce_block_size, impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 1, 3>(arg), reduce_block_size, impl);
                    return;
                case 2:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 2, 2>(arg), reduce_block_size, impl);
                    return;
                case 3:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 3, 1>(arg), reduce_block_size, impl);
                    return;
                case 4:
                    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
                            MakeReductionKernelArg<In, Out, 4, 4, 0>(arg), reduce_block_size, impl);
                    return;
            }
            XCHAINER_NEVER_REACH();
    }
#endif

    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(
            MakeReductionKernelArg<In, Out>(arg), reduce_block_size, impl);
}

}  // namespace cuda
}  // namespace xchainer
