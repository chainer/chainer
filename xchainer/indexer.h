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

class NdimIndex {
public:
    NdimIndex(const int64_t* index, int8_t ndim) : ndim_{ndim} { std::copy_n(index, ndim, index_); }

    explicit NdimIndex(int8_t ndim) : ndim_{ndim} {
        for (int8_t i = 0; i < ndim; ++i) {
            index_[i] = 0;
        }
    }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

private:
    int64_t index_[kMaxNdim];
    int8_t ndim_;
};

namespace indexer_detail {

// IndexSource is either IndexIterator or NdimIndex.
template <int8_t Ndim, typename IndexSource>
XCHAINER_HOST_DEVICE inline int8_t CombineIteratorsImplBase(
        IndexIterator<Ndim>& it, int8_t processed_dims, const IndexSource& index_source) {
    assert(processed_dims + index_source.ndim() <= it.ndim());
    for (int8_t i = 0; i < index_source.ndim(); ++i) {
        it.index()[processed_dims + i] = index_source.index()[i];
    }
    return processed_dims + index_source.ndim();
}

template <int8_t Ndim>
XCHAINER_HOST_DEVICE inline int8_t CombineIteratorsImpl(IndexIterator<Ndim>& /*it*/, int8_t processed_dims) {
    return processed_dims;
}

template <int8_t Ndim, typename IndexSource, typename... IndexSources>
XCHAINER_HOST_DEVICE inline int8_t CombineIteratorsImpl(
        IndexIterator<Ndim>& it, int8_t processed_dims, IndexSource&& index_source, IndexSources&&... index_sources) {
    processed_dims = CombineIteratorsImplBase(it, processed_dims, index_source);
    return CombineIteratorsImpl(it, processed_dims, std::forward<IndexSources>(index_sources)...);
}

// Combine multiple sub-iterators to make a combined iterator.
// Returns the number of written dimensions, which is equal to ndim_.
// `processed_dims` is the number of written dimensions so far.
template <int8_t Ndim, typename... IndexSources>
XCHAINER_HOST_DEVICE void CombineIterators(IndexIterator<Ndim>& it, IndexSources&&... index_sources) {
    int8_t processed_dims = indexer_detail::CombineIteratorsImpl<Ndim>(it, 0, std::forward<IndexSources>(index_sources)...);
    (void)processed_dims;  // unused
    assert(processed_dims == it.ndim());
    if (XCHAINER_DEBUG) {
        for (int8_t i = 0; i < it.ndim(); ++i) {
            assert(0 <= it.index()[i]);
        }
    }
}

}  // namespace indexer_detail

template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape.GetTotalSize()} {
        assert(shape.ndim() == kNdim);
        std::copy_n(shape.begin(), kNdim, shape_);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kNdim>{shape_, total_size_, start, step};
    }

    // Sets an index from multiple indexers each of which composes a portion of dimensions in order.
    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<kNdim> At(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
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

// Static 0-dimensional specialization.
template <>
class Indexer<0> {
public:
    explicit Indexer(const Shape& shape) {
        (void)shape;  // unused
        assert(shape.ndim() == 0);
        assert(shape.GetTotalSize() == 1);
    }

    XCHAINER_HOST_DEVICE IndexIterator<0> It(int64_t start, int64_t step = 1) const { return IndexIterator<0>{start, step}; }

    XCHAINER_HOST_DEVICE IndexIterator<0> At(const IndexIterator<0>& iter) { return It(iter.raw_index()); }

    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<0> At(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        IndexIterator<0> it = It(0);
        indexer_detail::CombineIterators<0>(it, first_iter, std::forward<IndexIterators>(iters)...);
        return it;
    }

    XCHAINER_HOST_DEVICE static constexpr int8_t ndim() { return 0; }

    XCHAINER_HOST_DEVICE static constexpr int64_t total_size() { return 1; }

    XCHAINER_HOST_DEVICE static constexpr const int64_t* shape() { return nullptr; }
};

// Static 1-dimensional specialization.
template <>
class Indexer<1> {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape[0]} { assert(1 == shape.ndim()); }

    XCHAINER_HOST_DEVICE IndexIterator<1> It(int64_t start, int64_t step = 1) const { return IndexIterator<1>{total_size_, start, step}; }

    XCHAINER_HOST_DEVICE IndexIterator<1> At(const IndexIterator<1>& iter) { return It(iter.raw_index()); }

    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE IndexIterator<1> At(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        IndexIterator<1> it = It(0);
        indexer_detail::CombineIterators<1>(it, first_iter, std::forward<IndexIterators>(iters)...);
        return it;
    }

    XCHAINER_HOST_DEVICE static constexpr int8_t ndim() { return 1; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return &total_size_; }

private:
    int64_t total_size_{};
};

// Runtime determined dynamic dimension specialization.
template <>
class Indexer<kDynamicNdim> {
public:
    explicit Indexer(const Shape& shape) : ndim_{shape.ndim()}, total_size_{shape.GetTotalSize()} {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kDynamicNdim>{shape_, ndim_, total_size_, start, step};
    }

    template <typename IndexSource, typename... IndexSources>
    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim> At(IndexSource&& index_source, IndexSources&&... index_sources) {
        IndexIterator<kDynamicNdim> it = It(0);
        indexer_detail::CombineIterators<kDynamicNdim>(
                it, std::forward<IndexSource>(index_source), std::forward<IndexSources>(index_sources)...);
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
