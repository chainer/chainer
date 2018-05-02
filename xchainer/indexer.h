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

template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape.GetTotalSize()} { std::copy(shape.begin(), shape.end(), shape_); }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kNdim>{shape_, total_size_, start, step};
    }

    // Sets an index from multiple indexers each of which composes a portion of dimensions in order.
    template <int8_t kNdimFirst, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest);

    XCHAINER_HOST_DEVICE int8_t ndim() const { return kNdim; }
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

    template <int8_t kNdimFirst, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> It(const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest);

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }
    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }
    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

private:
    int8_t ndim_{};
    int64_t total_size_{};
    int64_t shape_[kMaxNdim]{};
};

namespace internal {

template <int8_t kNdim, int8_t kNdimIter>
XCHAINER_HOST_DEVICE inline int8_t CombineIterators(IndexIterator<kNdim>& it, int8_t processed_dims, const IndexIterator<kNdimIter>& iter) {
    assert(processed_dims + iter.ndim() <= it.ndim());
    for (int8_t i = 0; i < iter.ndim(); ++i) {
        it.index()[processed_dims + i] = iter.index()[i];
    }
    return processed_dims + iter.ndim();
}

// Combine multiple sub-iterators to make a combined iterator.
// Returns the number of written dimensions, which is equal to ndim_.
// `processed_dims` is the number of written dimensions so far.
template <int8_t kNdim, int8_t kNdimFirstIter, typename... IndexIterators>
XCHAINER_HOST_DEVICE int8_t
CombineIterators(IndexIterator<kNdim>& it, int8_t processed_dims, const IndexIterator<kNdimFirstIter>& first, IndexIterators&&... rest) {
    processed_dims = CombineIterators(it, processed_dims, first);
    int8_t dims = CombineIterators(it, processed_dims, std::forward<IndexIterators>(rest)...);
    assert(dims == it.ndim());
    return dims;
}

template <int8_t kNdim, typename... IndexIterators>
XCHAINER_HOST_DEVICE IndexIterator<kNdim> CombinedIterator(IndexIterator<kNdim>& it, IndexIterators&&... rest) {
    int8_t processed_dims = internal::CombineIterators<kNdim>(it, 0, std::forward<IndexIterators>(rest)...);
    assert(processed_dims == it.ndim());
#ifndef NDEBUG
    for (int8_t i = 0; i < it.ndim(); ++i) {
        assert(0 <= it.index()[i]);
    }
#endif
    return it;
}

}  // namespace internal

// Sets an index from mutiple indexers each of which composes a portion of dimensions in order.
template <int8_t kNdim>
template <int8_t kNdimFirst, typename... IndexIterators>
XCHAINER_HOST_DEVICE IndexIterator<kNdim> Indexer<kNdim>::It(const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest) {
    auto it = It(0);
    return internal::CombinedIterator<kNdim>(it, first, std::forward<IndexIterators>(rest)...);
}

// Dynamic-length specialization.
template <int8_t kNdimFirst, typename... IndexIterators>
XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> Indexer<kDynamicNdim>::It(
        const IndexIterator<kNdimFirst>& first, IndexIterators&&... rest) {
    auto it = It(0);
    return internal::CombinedIterator<kDynamicNdim>(it, first, std::forward<IndexIterators>(rest)...);
}

template <int8_t kNdim>
inline std::ostream& operator<<(std::ostream& os, const Indexer<kNdim>& indexer) {
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << ")";
}

}  // namespace xchainer
