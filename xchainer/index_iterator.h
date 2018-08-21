#pragma once

#include <cassert>
#include <cstdint>
#include <utility>

#include "xchainer/constant.h"
#include "xchainer/macro.h"

namespace xchainer {

template <int8_t kNdim = kDynamicNdim>
class IndexIterator;

namespace index_iterator_detail {

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

}  // namespace index_iterator_detail

namespace internal {

// Combine multiple sub-iterators to make a combined iterator.
// Returns the number of written dimensions, which is equal to ndim_.
// `processed_dims` is the number of written dimensions so far.
//
// TODO(sonots): Set raw_index
template <int8_t Ndim, typename... IndexSources>
XCHAINER_HOST_DEVICE void CombineIterators(IndexIterator<Ndim>& it, IndexSources&&... index_sources) {
    int8_t processed_dims = index_iterator_detail::CombineIteratorsImpl<Ndim>(it, 0, std::forward<IndexSources>(index_sources)...);
    (void)processed_dims;  // unused
    assert(processed_dims == it.ndim());
#ifndef NDEBUG
    for (int8_t i = 0; i < it.ndim(); ++i) {
        assert(0 <= it.index()[i]);
    }
#endif  // NDEBUG
}

}  // namespace internal

template <int8_t kNdim>
class IndexIterator {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, total_size_{total_size}, raw_index_{0}, start_{start}, step_{step}, index_{} {
        assert(start >= 0);
        assert(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
        if (total_size > 0) {
            Set(start);
        }
    }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    XCHAINER_HOST_DEVICE IndexIterator<kNdim>& operator--() {
        Set(raw_index_ - step_);
        return *this;
    }

    XCHAINER_HOST_DEVICE void Restart() { Set(start_); }

    XCHAINER_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        start_ = start;
        if (total_size_ > 0) {
            Set(start_);
        }
    }

    // Sets an index from multiple indexers each of which composes a portion of dimensions in order.
    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE void Combine(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        internal::CombineIterators<kNdim>(*this, first_iter, std::forward<IndexIterators>(iters)...);
    }

    template <typename IndexSource>
    XCHAINER_HOST_DEVICE void CopyFrom(IndexSource index_source, int8_t offset = 0) {
        for (int i = 0; i < index_source.ndim(); ++i) {
            index_[i + offset] = index_source.index()[i];
        }
    }

    XCHAINER_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }

    XCHAINER_HOST_DEVICE constexpr int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        assert(total_size_ > 0);
        raw_index_ = i;
        for (int8_t j = kNdim; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

    const int64_t* shape_;
    int64_t total_size_{};
    int64_t raw_index_{};
    int64_t start_{};
    int64_t step_{};
    int64_t index_[kNdim];
};

// Static 0-dimensional specialization.
template <>
class IndexIterator<0> {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : IndexIterator<0>{start, step} {
        (void)shape;  // unused
        (void)total_size;  // unused
        assert(total_size == 1);
    }

    explicit XCHAINER_HOST_DEVICE IndexIterator(int64_t start, int64_t step) : raw_index_{start} {
        (void)step;  // unused
        assert(start >= 0);
        assert(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
    }

    XCHAINER_HOST_DEVICE IndexIterator<0>& operator++() {
        ++raw_index_;
        return *this;
    }

    XCHAINER_HOST_DEVICE IndexIterator<0>& operator--() {
        --raw_index_;
        return *this;
    }

    XCHAINER_HOST_DEVICE void Restart() { raw_index_ = 0; }

    XCHAINER_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        raw_index_ = start;
    }

    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE void Combine(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        internal::CombineIterators<0>(*this, first_iter, std::forward<IndexIterators>(iters)...);
    }

    template <typename IndexSource>
    XCHAINER_HOST_DEVICE void CopyFrom(IndexSource index_source, int8_t offset = 0) {
        (void)index_source; // unused;
        (void)offset;  // unused
        assert(index_source.ndim() == 0);
        assert(offset == 0);
    }

    XCHAINER_HOST_DEVICE operator bool() const { return raw_index_ < 1; }

    XCHAINER_HOST_DEVICE static constexpr int8_t ndim() { return 0; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return &raw_index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return &raw_index_; }

private:
    int64_t raw_index_{0};
};

// Static 1-dimensional specialization.
template <>
class IndexIterator<1> {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : IndexIterator<1>{total_size, start, step} {
        assert(shape[0] == total_size);
        (void)shape;  // unused, except for sanity check.
    }

    explicit XCHAINER_HOST_DEVICE IndexIterator(int64_t total_size, int64_t start, int64_t step)
        : total_size_{total_size}, raw_index_{start}, start_{start}, step_{step} {
        assert(start >= 0);
        assert(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
    }

    XCHAINER_HOST_DEVICE IndexIterator<1>& operator++() {
        raw_index_ += step_;
        return *this;
    }

    XCHAINER_HOST_DEVICE IndexIterator<1>& operator--() {
        raw_index_ -= step_;
        return *this;
    }

    XCHAINER_HOST_DEVICE void Restart() { raw_index_ = start_; }

    XCHAINER_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        raw_index_ = start;
    }

    template <int8_t NdimArg, typename... IndexIterators>
    XCHAINER_HOST_DEVICE void Combine(const IndexIterator<NdimArg>& first_iter, IndexIterators&&... iters) {
        internal::CombineIterators<1>(*this, first_iter, std::forward<IndexIterators>(iters)...);
    }

    template <typename IndexSource>
    XCHAINER_HOST_DEVICE void CopyFrom(IndexSource index_source, int8_t offset = 0) {
        (void)index_source;  // unused
        assert(index_source.ndim() == 1);
        assert(offset == 0);
        raw_index_ = index_source.index()[0];
    }

    XCHAINER_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }

    XCHAINER_HOST_DEVICE static constexpr int8_t ndim() { return 1; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return &raw_index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return &raw_index_; }

private:
    int64_t total_size_{};
    int64_t raw_index_{};
    int64_t start_{};
    int64_t step_{};
};

// Runtime determined dynamic dimension specialization.
template <>
class IndexIterator<kDynamicNdim> {
public:
    explicit XCHAINER_HOST_DEVICE IndexIterator(const int64_t* shape, int8_t ndim, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, ndim_{ndim}, total_size_{total_size}, raw_index_{0}, start_{start}, step_{step}, index_{} {
        assert(start >= 0);
        assert(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
        if (total_size > 0) {
            Set(start);
        }
    }

    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    XCHAINER_HOST_DEVICE IndexIterator<kDynamicNdim>& operator--() {
        Set(raw_index_ - step_);
        return *this;
    }

    XCHAINER_HOST_DEVICE void Restart() { Set(start_); }

    XCHAINER_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        start_ = start;
        if (total_size_ > 0) {
            Set(start_);
        }
    }

    template <typename IndexSource, typename... IndexSources>
    XCHAINER_HOST_DEVICE void Combine(IndexSource&& index_source, IndexSources&&... index_sources) {
        internal::CombineIterators<kDynamicNdim>(
                *this, std::forward<IndexSource>(index_source), std::forward<IndexSources>(index_sources)...);
    }

    template <typename IndexSource>
    XCHAINER_HOST_DEVICE void CopyFrom(IndexSource index_source, int8_t offset = 0) {
        for (int i = 0; i < index_source.ndim(); ++i) {
            index_[i + offset] = index_source.index()[i];
        }
    }

    XCHAINER_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        assert(total_size_ > 0);
        raw_index_ = i;
        for (int8_t j = ndim_; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

    const int64_t* shape_;
    int8_t ndim_{};
    int64_t total_size_{};
    int64_t raw_index_{};
    int64_t start_{};
    int64_t step_{};
    int64_t index_[kMaxNdim];
};

}  // namespace xchainer
