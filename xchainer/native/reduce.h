#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/reduction_kernel_arg.h"

namespace xchainer {
namespace native {
namespace reduce_detail {

template <
        typename In,
        typename Out,
        typename ReductionImpl,
        int8_t InNdim = kDynamicNdim,
        int8_t OutNdim = kDynamicNdim,
        int8_t ReduceNdim = kDynamicNdim>
void ReductionKernel(ReductionKernelIndexableArg<In, Out, InNdim, OutNdim, ReduceNdim> arg, ReductionImpl&& impl) {
    auto it_in = arg.in_indexer.It(0);

    // Iterate over output dimensions
    for (auto it_out = arg.out_indexer.It(0); it_out; ++it_out) {
        auto accum = impl.Identity();

        // Set output indices in the corresponding indices (out_axis) in src_index.
        for (int8_t i_out_dim = 0; i_out_dim < arg.out_indexer.ndim(); ++i_out_dim) {
            it_in.index()[i_out_dim] = it_out.index()[i_out_dim];
        }

        // Iterate over reduction dimensions, reducing into a single output value.
        for (auto it_reduce = arg.reduce_indexer.It(0); it_reduce; ++it_reduce) {
            // Set reduction indices in the corresponding indices (axis) in src_index.
            for (int8_t i_reduce_dim = 0; i_reduce_dim < arg.reduce_indexer.ndim(); ++i_reduce_dim) {
                it_in.index()[arg.out_indexer.ndim() + i_reduce_dim] = it_reduce.index()[i_reduce_dim];
            }

            impl.Reduce(impl.MapIn(arg.in[it_in], it_reduce.raw_index()), accum);
        }
        arg.out[it_out] = impl.MapOut(accum);
    }
}

}  // namespace reduce_detail

// Computes the reduction of the input and stores into the output array.
//
// `ReductionImpl` is required to provide the following member function.
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
//             float Identity() { return 0; }
//             float MapIn(float in) { return in; }
//             void Reduce(float next, float& accum) { accum += next; }
//             float MapOut(float accum) { return accum; }
//         };
//
//     Then, it can be passed to Reduce like: Reduce(MakeReductionKernelArg(input, axis, output), SumImpl{});
template <typename In, typename Out, typename ReductionImpl>
void Reduce(ReductionKernelArg<In, Out> arg, ReductionImpl&& impl) {
    if (arg.in_strides.ndim() == 1 && arg.out_strides.ndim() == 0) {
        reduce_detail::ReductionKernel(MakeReductionKernelIndexableArg<In, Out, 1, 0, 1>(arg), impl);
        return;
    }
    if (arg.in_strides.ndim() == 2 && arg.out_strides.ndim() == 1) {
        reduce_detail::ReductionKernel(MakeReductionKernelIndexableArg<In, Out, 2, 1, 1>(arg), impl);
        return;
    }
    reduce_detail::ReductionKernel(MakeReductionKernelIndexableArg<In, Out>(arg), impl);
}

}  // namespace native
}  // namespace xchainer
