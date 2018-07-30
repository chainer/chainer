#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/reduction_kernel_arg.h"

namespace xchainer {

template <typename In, typename Out, int8_t InNdim, int8_t OutNdim, int8_t ReduceNdim>
struct ReductionKernelArg2 {
    IndexableArray<const In, InNdim> in;
    IndexableArray<Out, OutNdim> out;
    Indexer<InNdim> in_indexer;
    Indexer<OutNdim> out_indexer;
    Indexer<ReduceNdim> reduce_indexer;
};

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

template <typename In, typename Out, int8_t InNdim, int8_t OutNdim, int8_t ReduceNdim, typename ReductionImpl>
__global__ void ReductionKernel(ReductionKernelArg2<In, Out, InNdim, OutNdim, ReduceNdim> arg, int reduce_block_size, ReductionImpl impl) {
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
//     Then, it can be passed to Reduce like: Reduce(MakeReductionKernelArg(input, axis, output), SumImpl{});
template <typename In, typename Out, int8_t InNdim, int8_t OutNdim, int8_t ReduceNdim, typename ReductionImpl>
void Reduce(ReductionKernelArg2<In, Out, InNdim, OutNdim, ReduceNdim> arg, ReductionImpl&& impl) {
    static const int kMaxBlockSize =
            CudaOccupancyMaxPotentialBlockSize(&reduce_detail::ReductionKernel<In, Out, InNdim, OutNdim, ReduceNdim, ReductionImpl>)
                    .block_size;

    int reduce_block_size = static_cast<int>(std::min(
            static_cast<int64_t>(reduce_detail::kMaxReductionBlockSize),
            reduce_detail::RoundUpToPowerOf2(std::max(int64_t{1}, arg.reduce_indexer.total_size()))));
    int block_size = std::min(kMaxBlockSize, reduce_block_size);
    int64_t total_reduce_blocks = arg.out_indexer.total_size();
    int64_t grid_size = total_reduce_blocks;
    size_t shared_mem_size = sizeof(decltype(impl.Identity())) * reduce_block_size;

    reduce_detail::ReductionKernel<<<grid_size, block_size, shared_mem_size>>>(arg, reduce_block_size, impl);
}

template <typename In, typename Out, typename ReductionImpl>
void LaunchReductionKernel(const Array& in, const Axes& axis, const Array& out, ReductionImpl&& impl) {
    // True if some axes are reduced but kept in output as 1-dim axes.
    // Corresponding to keepdim argument in Array::Sum().
    bool has_kept_dims = out.ndim() + static_cast<int64_t>(axis.size()) != in.ndim();

    // Prepare axis mappings
    Shape reduce_shape{};  // Reduction dimensions
    Axes out_axis_map{};  // Mapping from effective output indices to actual output indices
    Shape new_out_shape{};
    // (Here "effective output indices" means source indices minus reduction indices.)

    // Example (in the case of has_kept_dims=false):
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 14, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 1, 2)
    // - new_out_shape:    (12, 14, 16)
    // - new_in_shape:     (12, 14, 16, 13, 15)
    // Example (in the case of has_kept_dims=true):
    // - in.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 1, 14, 1, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 2, 4)
    // - new_out_shape:    (12, 14, 16)
    // - new_in_shape:     (12, 14, 16, 13, 15)

    {
        size_t i_axis = 0;
        size_t i_out_axis = 0;
        for (int8_t i = 0; i < in.shape().ndim(); ++i) {
            if (i_axis < axis.size() && i == axis[i_axis]) {
                // i is to be reduced
                int64_t in_dim = in.shape()[i];
                if (in_dim != 1) {
                    reduce_shape.emplace_back(in_dim);
                }
                ++i_axis;
                if (has_kept_dims) {
                    ++i_out_axis;
                }
            } else {
                // i is not to be reduced
                int64_t out_dim = out.shape()[i_out_axis];
                if (out_dim != 1) {
                    out_axis_map.emplace_back(static_cast<int8_t>(i_out_axis));
                    new_out_shape.emplace_back(out_dim);
                }
                ++i_out_axis;
            }
        }
        assert(i_out_axis == out.shape().size());
        assert(i_axis == axis.size());
    }
    // Inequality because 1-dim axes are eliminated.
    assert(reduce_shape.size() <= axis.size());
    assert(out_axis_map.size() <= in.shape().size() - axis.size());
    assert(out_axis_map.size() == new_out_shape.size());

    // Calculate source axis permutation
    Axes axis_permutes{};
    {
        size_t i_reduce = 0;
        for (int8_t i = 0; i < in.ndim(); ++i) {
            if (i_reduce < axis.size() && i == axis[i_reduce]) {
                ++i_reduce;
            } else {
                if (in.shape()[i] != 1) {
                    axis_permutes.emplace_back(i);
                }
            }
        }
    }
    for (int8_t i : axis) {
        if (in.shape()[i] != 1) {
            axis_permutes.emplace_back(i);
        }
    }
    assert(axis_permutes.size() <= in.shape().size());  // Inequality because 1-dim axes are eliminated.

    // Calculate new source shape
    Shape new_in_shape{};
    for (int8_t i : axis_permutes) {
        new_in_shape.emplace_back(in.shape()[i]);
    }

    // 1-dim axes must be eliminated
    assert(std::find(new_in_shape.begin(), new_in_shape.end(), 1) == new_in_shape.end());
    assert(std::find(new_out_shape.begin(), new_out_shape.end(), 1) == new_out_shape.end());

    if (in.ndim() == 1 && out.ndim() == 0) {
        return Reduce<In, Out, 1, 0, 1>(
                ReductionKernelArg2<In, Out, 1, 0, 1>{IndexableArray<const In, 1>{in}.Permute(axis_permutes),
                                                      IndexableArray<Out, 0>{out}.Permute(out_axis_map),
                                                      Indexer<1>{new_in_shape},
                                                      Indexer<0>{new_out_shape},
                                                      Indexer<1>{reduce_shape}},
                impl);
    }
    return Reduce<In, Out, kDynamicNdim, kDynamicNdim, kDynamicNdim>(
            ReductionKernelArg2<In, Out, kDynamicNdim, kDynamicNdim, kDynamicNdim>{
                    IndexableArray<const In, kDynamicNdim>{in}.Permute(axis_permutes),
                    IndexableArray<Out, kDynamicNdim>{out}.Permute(out_axis_map),
                    Indexer<kDynamicNdim>{new_in_shape},
                    Indexer<kDynamicNdim>{new_out_shape},
                    Indexer<kDynamicNdim>{reduce_shape}},
            impl);
}

}  // namespace cuda
}  // namespace xchainer
