#pragma once

#include <cstdint>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/reduction_kernel_arg.h"

namespace chainerx {
namespace native {
namespace reduce_detail {

constexpr int64_t ExpandLen = 4;
constexpr int64_t SerialLen = 8;

template <typename In, typename ReductionImpl, int8_t InNdim, typename T, int64_t n>
struct ExpandedPairwiseReduction {
    static T run(const IndexableArray<const In, InNdim>& in, IndexIterator<InNdim>& it_in, ReductionImpl&& impl, int64_t& i_reduce) {
        T accum = ExpandedPairwiseReduction<In, ReductionImpl, InNdim, T, n / 2>::run(in, it_in, impl, i_reduce);
        impl.Reduce(ExpandedPairwiseReduction<In, ReductionImpl, InNdim, T, n / 2>::run(in, it_in, impl, i_reduce), accum);
        return accum;
    }
};

template <typename In, typename ReductionImpl, int8_t InNdim, typename T>
struct ExpandedPairwiseReduction<In, ReductionImpl, InNdim, T, 1> {
    static T run(const IndexableArray<const In, InNdim>& in, IndexIterator<InNdim>& it_in, ReductionImpl&& impl, int64_t& i_reduce) {
        T accum = impl.MapIn(native_internal::StorageToDataType<const In>(in[it_in]), i_reduce);
        ++it_in, ++i_reduce;
        return accum;
    }
};

inline int64_t pairwise_len(int64_t reduce_len) { return ((reduce_len / ExpandLen + (SerialLen - 1)) / SerialLen); }

template <typename In, typename ReductionImpl, int8_t InNdim, typename T>
T PairwiseReduction(
        const IndexableArray<const In, InNdim>& in,
        IndexIterator<InNdim>& it_in,
        ReductionImpl&& impl,
        int64_t reduce_len,
        std::vector<T>& tree_accum) {
    int64_t i_reduce = 0;
    T accum = impl.Identity();

    bool first_loop = true;
    while (i_reduce < (reduce_len & -ExpandLen)) {
        if (first_loop) {
            first_loop = false;
        } else if ((i_reduce & (SerialLen * ExpandLen - 1)) == 0) {
            int i = 0;
            for (int64_t k = i_reduce >> 1; (k & (SerialLen * ExpandLen - 1)) == 0; k >>= 1, ++i) {
                impl.Reduce(tree_accum[i], accum);
            }
            tree_accum[i] = accum;
            accum = impl.Identity();
        }
        impl.Reduce(ExpandedPairwiseReduction<In, ReductionImpl, InNdim, T, ExpandLen>::run(in, it_in, impl, i_reduce), accum);
    }

    while (i_reduce < reduce_len) {
        impl.Reduce(impl.MapIn(native_internal::StorageToDataType<const In>(in[it_in]), i_reduce), accum);
        ++it_in, ++i_reduce;
    }

    int64_t k = pairwise_len(reduce_len) - 1;
    for (const T& leaf_accum : tree_accum) {
        if (k & 1) {
            impl.Reduce(leaf_accum, accum);
        }
        k >>= 1;
    }
    return accum;
}

inline int bits_of_index(int64_t n) {
    if (--n <= 0) return 0;
    int64_t t;
    int bits = 1;
    if ((t = n >> 32)) bits += 32, n = t;
    if ((t = n >> 16)) bits += 16, n = t;
    if ((t = n >> 8)) bits += 8, n = t;
    if ((t = n >> 4)) bits += 4, n = t;
    if ((t = n >> 2)) bits += 2, n = t;
    bits += static_cast<int>(n >> 1);
    return bits;
}

template <typename In, typename Out, typename ReductionImpl, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim>
void ReductionKernel(ReductionKernelArg<In, Out, InNdim, OutNdim> arg, ReductionImpl&& impl) {
    auto it_in = arg.in_indexer.It(0, arg.out_indexer.total_size());
    int64_t reduce_len = arg.in_indexer.total_size() / arg.out_indexer.total_size();
    std::vector<decltype(impl.Identity())> tree_accum(bits_of_index(pairwise_len(reduce_len)), impl.Identity());

    // Iterate over output dimensions
    for (auto it_out = arg.out_indexer.It(0); it_out; ++it_out) {
        it_in.Restart(it_out.raw_index());
        auto accum = PairwiseReduction<In, ReductionImpl, InNdim, decltype(impl.Identity())>(arg.in, it_in, impl, reduce_len, tree_accum);
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
