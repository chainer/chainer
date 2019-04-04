#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/reduction_kernel_arg.h"

namespace chainerx {
namespace native {
namespace reduce_detail {

template <typename In, typename Out, typename ReductionImpl, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim>
void ReductionKernel(ReductionKernelArg<In, Out, InNdim, OutNdim> arg, ReductionImpl&& impl) {
    auto it_in = arg.in_indexer.It(0, arg.out_indexer.total_size());

    // Iterate over output dimensions
    for (auto it_out = arg.out_indexer.It(0); it_out; ++it_out) {
        auto accum = impl.Identity();

        int64_t i_reduce{0};
        for (it_in.Restart(it_out.raw_index()); it_in; ++it_in, ++i_reduce) {
            impl.Reduce(impl.MapIn(native_internal::StorageToDataType<const In>(arg.in[it_in]), i_reduce), accum);
        }

        arg.out[it_out] = native_internal::DataToStorageType<Out>(impl.MapOut(accum));
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
//     Then, it can be passed to Reduce like: Reduce(input, axis, output, SumImpl{});
template <typename In, typename Out, typename ReductionImpl>
void Reduce(const Array& in, const Axes& axis, const Array& out, ReductionImpl&& impl) {
    if (out.GetTotalSize() == 0) {
        return;
    }

    ReductionArg arg{in, axis, out};

    // TODO(sonots): Reconsider the number of statically-optimized kernels in terms of speed and binary size trade-offs.
    // Currently, we optimize for contiguous output arrays.
    switch (arg.in_shape().ndim()) {
        case 1:
            switch (arg.out_shape().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 1, 0>(arg), impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 1, 1>(arg), impl);
                    return;
            }
            break;
        case 2:
            switch (arg.out_shape().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 2, 0>(arg), impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 2, 1>(arg), impl);
                    return;
            }
            break;
        case 3:
            switch (arg.out_shape().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 3, 0>(arg), impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 3, 1>(arg), impl);
                    return;
            }
            break;
        case 4:
            switch (arg.out_shape().ndim()) {
                case 0:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 4, 0>(arg), impl);
                    return;
                case 1:
                    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out, 4, 1>(arg), impl);
                    return;
            }
            break;
    }

    reduce_detail::ReductionKernel(MakeReductionKernelArg<In, Out>(arg), impl);
}

}  // namespace native
}  // namespace chainerx
