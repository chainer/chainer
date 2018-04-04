#pragma once

#include <cstdint>

#include "xchainer/array.h"
#include "xchainer/reduction_kernel_arg.h"

namespace xchainer {
namespace native {
namespace internal {

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
//     Then, it can be passed to Reduce like: Reduce(MakeReductionArg(input, axis, output), SumImpl{});
template <typename In, typename Out, typename ReductionImpl>
void Reduce(ReductionArg<In, Out> arg, ReductionImpl&& impl) {
    // Iterate over output dimensions
    for (int64_t i_out = 0; i_out < arg.out_indexer.total_size(); ++i_out) {
        arg.out_indexer.Set(i_out);

        auto accum = impl.Identity();

        // Set output indices in the corresponding indices (out_axis) in src_index.
        for (int8_t i_out_dim = 0; i_out_dim < arg.out_indexer.ndim(); ++i_out_dim) {
            arg.in_indexer.index()[i_out_dim] = arg.out_indexer.index()[i_out_dim];
        }

        // Iterate over reduction dimensions, reducing into a single output value.
        for (int64_t i_reduce = 0; i_reduce < arg.reduce_indexer.total_size(); ++i_reduce) {
            arg.reduce_indexer.Set(i_reduce);
            // Set reduction indices in the corresponding indices (axis) in src_index.
            for (int8_t i_reduce_dim = 0; i_reduce_dim < arg.reduce_indexer.ndim(); ++i_reduce_dim) {
                arg.in_indexer.index()[arg.out_indexer.ndim() + i_reduce_dim] = arg.reduce_indexer.index()[i_reduce_dim];
            }

            impl.Reduce(impl.MapIn(arg.in[arg.in_indexer], i_reduce), accum);
        }
        arg.out[arg.out_indexer] = impl.MapOut(accum);
    }
}

}  // namespace internal
}  // namespace native
}  // namespace xchainer
