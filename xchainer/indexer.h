#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ostream>

#include "xchainer/constant.h"
#include "xchainer/macro.h"
#include "xchainer/shape.h"

namespace xchainer {

class IndexIterator;

template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape.GetTotalSize()}, ndim_{shape.ndim()} {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE IndexIterator It(int64_t start, int64_t step = 1) const;

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    // Sets an index from mutiple indexers each of which composes a portion of dimensions in order.
    template <typename... Args>
    XCHAINER_HOST_DEVICE IndexIterator It(const IndexIterator& first, Args&&... iters);

private:
    // Combine multiple sub-iterators to make a combined iterator.
    // Returns the number of written dimensions, which is equal to ndim_.
    // `processed_dims` is the number of written dimensions so far.
    template <typename... Args>
    XCHAINER_HOST_DEVICE int8_t
    CombineIterators(IndexIterator& it, int8_t processed_dims, const IndexIterator& first_iter, Args&&... iters);

    XCHAINER_HOST_DEVICE int8_t CombineIterators(IndexIterator& it, int8_t processed_dims, const IndexIterator& iter);

private:
    int64_t shape_[kMaxNdim]{};
    int64_t total_size_{};
    int8_t ndim_{};
};

inline std::ostream& operator<<(std::ostream& os, const Indexer<>& indexer) {
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << ")";
}

class IndexIterator {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const Indexer<>& indexer, int64_t start, int64_t step)
        : indexer_{indexer}, index_{}, raw_index_{0}, step_{step} {
        // backward iteration is not supported in order to omit lower-bound check for performance.
        assert(start >= 0);
        assert(step > 0);
        Set(start);
    }

    XCHAINER_HOST_DEVICE const Indexer<>& indexer() const { return indexer_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE operator bool() const { return ok(); }

    XCHAINER_HOST_DEVICE IndexIterator& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

private:
    XCHAINER_HOST_DEVICE bool ok() const { return raw_index_ < indexer_.total_size(); }

    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        if (indexer_.total_size() == 0) {
            // In this case there are some j such that shape[j] == 0.
            return;
        }
        const int64_t* shape = indexer_.shape();
        for (int8_t j = indexer_.ndim(); --j >= 0;) {
            index_[j] = i % shape[j];
            i /= shape[j];
        }
    }

    const Indexer<>& indexer_;
    int64_t index_[kMaxNdim];
    int64_t raw_index_;
    const int64_t step_;
};

template <int8_t kNdim>
XCHAINER_HOST_DEVICE inline IndexIterator Indexer<kNdim>::It(int64_t start, int64_t step) const {
    return IndexIterator{*this, start, step};
}

// Sets an index from mutiple indexers each of which composes a portion of dimensions in order.
template <int8_t kNdim>
template <typename... Args>
XCHAINER_HOST_DEVICE IndexIterator Indexer<kNdim>::It(const IndexIterator& first, Args&&... iters) {
    IndexIterator it = It(0);
    int8_t processed_dims = CombineIterators(it, 0, first, std::forward<Args>(iters)...);
    assert(processed_dims == ndim_);
#ifndef NDEBUG
    for (int8_t i = 0; i < ndim_; ++i) {
        assert(0 <= it.index()[i]);
        assert(it.index()[i] < shape_[i]);
    }
#endif
    return it;
}

template <int8_t kNdim>
template <typename... Args>
XCHAINER_HOST_DEVICE int8_t
Indexer<kNdim>::CombineIterators(IndexIterator& it, int8_t processed_dims, const IndexIterator& first_iter, Args&&... iters) {
    processed_dims = CombineIterators(it, processed_dims, first_iter);
    int8_t dims = CombineIterators(it, processed_dims, std::forward<Args>(iters)...);
    assert(dims == ndim_);
    return dims;
}

template <int8_t kNdim>
XCHAINER_HOST_DEVICE inline int8_t Indexer<kNdim>::CombineIterators(IndexIterator& it, int8_t processed_dims, const IndexIterator& iter) {
    assert(processed_dims + iter.indexer().ndim_ <= ndim_);
    for (int8_t i = 0; i < iter.indexer().ndim_; ++i) {
        it.index()[processed_dims + i] = iter.index()[i];
    }
    return processed_dims + iter.indexer().ndim_;
}

}  // namespace xchainer
