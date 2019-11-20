#pragma once

#include <array>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/reduction_kernel_arg.h"

namespace chainerx {
namespace native {
namespace reduce_detail {

// The number of items those are processed by the statically expanded pairwise reduction routine.
// Reduction performance is sensitive to this parameter. Increasing the parameter may improve performance,
// but it can also severely degrade the performance due to register pressure.
// Must be a power of 2.
constexpr int64_t ExpandLen = 4;
// The number of items those are reduced by the serial loop. The loop runs on the outside of the statically expanded
// reduction. A large value makes overhead for pairwise reduction small. So increasing this parameter will improve
// performance a little, in exchange for numerical precision.
// Must be a power of 2.
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

constexpr int log2(int64_t v) { return v == 1 ? 0 : log2(v >> 1) + 1; }

template <typename In, typename ReductionImpl, int8_t InNdim, typename T>
T PairwiseReduction(const IndexableArray<const In, InNdim>& in, IndexIterator<InNdim>& it_in, ReductionImpl&& impl, int64_t reduce_len) {
    int64_t i_reduce = 0;
    T accum = impl.Identity();

    constexpr int TreeDepth = 63 - log2(ExpandLen * SerialLen);
    std::array<T, TreeDepth> tree_accum;  // NOLINT(cppcoreguidelines-pro-type-member-init)

    while (i_reduce < reduce_len - (reduce_len % ExpandLen)) {
        // Invoke dynamic pairwise reduction if `i_reduce` is multiple of `SerialLen * ExpandLen`.
        if (i_reduce != 0 && i_reduce % (SerialLen * ExpandLen) == 0) {
            int i = 0;
            for (int64_t k = i_reduce >> 1; k % (SerialLen * ExpandLen) == 0; k >>= 1, ++i) {
                impl.Reduce(tree_accum[i], accum);  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            }
            tree_accum[i] = accum;  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            accum = impl.Identity();
        }
        // This increments `i_reduce` by `ExpandLen`.
        impl.Reduce(ExpandedPairwiseReduction<In, ReductionImpl, InNdim, T, ExpandLen>::run(in, it_in, impl, i_reduce), accum);
    }

    // Accumulate residuals.
    while (i_reduce < reduce_len) {
        impl.Reduce(impl.MapIn(native_internal::StorageToDataType<const In>(in[it_in]), i_reduce), accum);
        ++it_in, ++i_reduce;
    }

    // Accumulate tree nodes.
    int i = 0;
    for (int64_t k = (reduce_len / ExpandLen - 1) / SerialLen; k > 0; k >>= 1, ++i) {
        if (k & 1) {
            impl.Reduce(tree_accum[i], accum);  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }

    return accum;
}

template <typename In, typename Out, typename ReductionImpl, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim>
void ReductionKernel(ReductionKernelArg<In, Out, InNdim, OutNdim> arg, ReductionImpl&& impl) {
    auto it_in = arg.in_indexer.It(0, arg.out_indexer.total_size());
    int64_t reduce_len = arg.in_indexer.total_size() / arg.out_indexer.total_size();

    // Iterate over output dimensions
    for (auto it_out = arg.out_indexer.It(0); it_out; ++it_out) {
        it_in.Restart(it_out.raw_index());
        auto accum = PairwiseReduction<In, ReductionImpl, InNdim, decltype(impl.Identity())>(arg.in, it_in, impl, reduce_len);
        arg.out[it_out] = native_internal::DataToStorageType<Out>(impl.MapOut(accum));
    }
}

template <typename In, typename Out, typename ReductionImpl, int8_t InNdim = kDynamicNdim, int8_t OutNdim = kDynamicNdim>
void ScanKernel(ReductionKernelArg<In, Out, InNdim, OutNdim> arg, ReductionImpl&& impl, int64_t reduce_len) {
    int64_t len = arg.in_indexer.total_size() / reduce_len;
    auto it_in = arg.in_indexer.It(0, len);
    auto it_out = arg.out_indexer.It(0, len);
    for (int64_t i = 0; i < len; ++i) {
        it_in.Restart(i);
        it_out.Restart(i);
        auto accum = impl.Identity();
        for (int64_t j = 0; j < reduce_len; ++j, ++it_in, ++it_out) {
            auto in = native_internal::StorageToDataType<const In>(arg.in[it_in]);
            impl.Reduce(impl.MapIn(in, i), accum);
            arg.out[it_out] = native_internal::DataToStorageType<Out>(impl.MapOut(accum));
        }
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

template <typename In, typename Out, typename ReductionImpl>
void Scan(const Array& in, int8_t axis, const Array& out, ReductionImpl&& impl) {
    if (out.GetTotalSize() == 0) {
        return;
    }

    ReductionArg arg{in, Axes{axis}, out};
    int64_t reduce_len = in.shape()[axis];

    if (arg.in_shape().ndim() == 1 && arg.out_shape().ndim() == 1) {
        reduce_detail::ScanKernel(MakeReductionKernelArg<In, Out, 1, 1>(arg), impl, reduce_len);
        return;
    }
    reduce_detail::ScanKernel(MakeReductionKernelArg<In, Out>(arg), impl, reduce_len);
}

}  // namespace native
}  // namespace chainerx
