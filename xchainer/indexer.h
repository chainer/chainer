#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ostream>

#include "xchainer/constant.h"
#include "xchainer/index_iterator.h"
#include "xchainer/macro.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace indexer_detail {

template <int8_t Ndim, int8_t NdimArg>
XCHAINER_HOST_DEVICE inline int8_t CombineIteratorsImpl(
        IndexIterator<Ndim>& it, int8_t processed_dims, const IndexIterator<NdimArg>& iter) {
    assert(processed_dims + iter.ndim() <= it.ndim());
    for (int8_t i = 0; i < iter.ndim(); ++i) {
        it.index()[processed_dims + i] = iter.index()[i];
    }
    return processed_dims + iter.ndim();
}

template <int8_t Ndim, int8_t NdimArg, typename... IndexIterators>
XCHAINER_HOST_DEVICE inline int8_t CombineIteratorsImpl(
        IndexIterator<Ndim>& it, int8_t processed_dims, const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
    processed_dims = CombineIteratorsImpl(it, processed_dims, first_iter);
    return CombineIteratorsImpl(it, processed_dims, std::forward<IndexIterators>(iters)...);
}

// Combine multiple sub-iterators to make a combined iterator.
// Returns the number of written dimensions, which is equal to ndim_.
// `processed_dims` is the number of written dimensions so far.
template <int8_t Ndim, typename... IndexIterators>
XCHAINER_HOST_DEVICE void CombineIterators(IndexIterator<Ndim>& it, IndexIterators&&... iters) {
    int8_t processed_dims = indexer_detail::CombineIteratorsImpl<Ndim>(it, 0, std::forward<IndexIterators>(iters)...);
    assert(processed_dims == it.ndim());
#ifndef NDEBUG
    for (int8_t i = 0; i < it.ndim(); ++i) {
        assert(0 <= it.index()[i]);
    }
#endif
}

}  // namespace indexer_detail

template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape.GetTotalSize()} {
        assert(shape.ndim() == kNdim);
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kNdim>{shape_, total_size_, start, step};
    }

    // Sets an index from multiple indexers each of which composes a portion of dimensions in order.
    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        IndexIterator<kNdim> it = It(0);
        indexer_detail::CombineIterators<kNdim>(it, first_iter, std::forward<IndexIterators>(iters)...);
        return it;
    }

    XCHAINER_HOST_DEVICE constexpr int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

private:
    int64_t total_size_{};
    int64_t shape_[kNdim]{};
};

// Dynamic-length specialization.
template <>
class Indexer<kDynamicNdim> {
public:
    explicit Indexer(const Shape& shape) : ndim_{shape.ndim()}, total_size_{shape.GetTotalSize()} {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kDynamicNdim>{shape_, ndim_, total_size_, start, step};
    }

    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> It(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        IndexIterator<kDynamicNdim> it = It(0);
        indexer_detail::CombineIterators<kDynamicNdim>(it, first_iter, std::forward<IndexIterators>(iters)...);
        return it;
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

private:
    int8_t ndim_{};
    int64_t total_size_{};
    int64_t shape_[kMaxNdim]{};
};

template <int8_t Ndim>
inline std::ostream& operator<<(std::ostream& os, const Indexer<Ndim>& indexer) {
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << ")";
}

}  // namespace xchainer
