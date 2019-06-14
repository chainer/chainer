#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>

#include <gsl/gsl>

#include "chainerx/constant.h"
#include "chainerx/index_iterator.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"

namespace chainerx {

class NdimIndex {
public:
    // Suppressing error with the following line that `index_` is not initialized.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    NdimIndex(const int64_t* index, int8_t ndim) : ndim_{ndim} { std::copy_n(index, ndim, index_); }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    explicit NdimIndex(int8_t ndim) : ndim_{ndim} {
        for (int8_t i = 0; i < ndim; ++i) {
            gsl::at(index_, i) = 0;
        }
    }

    CHAINERX_HOST_DEVICE int64_t* index() { return index_; }

    CHAINERX_HOST_DEVICE const int64_t* index() const { return index_; }

    CHAINERX_HOST_DEVICE int8_t ndim() const { return ndim_; }

private:
    int64_t index_[kMaxNdim];
    int8_t ndim_;
};

template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape.GetTotalSize()} {
        CHAINERX_ASSERT(shape.ndim() == kNdim);
        std::copy_n(shape.begin(), kNdim, shape_);
    }

    CHAINERX_HOST_DEVICE IndexIterator<kNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kNdim>{shape_, total_size_, start, step};
    }

    CHAINERX_HOST_DEVICE constexpr int8_t ndim() const { return kNdim; }

    CHAINERX_HOST_DEVICE int64_t total_size() const { return total_size_; }

    CHAINERX_HOST_DEVICE const int64_t* shape() const { return shape_; }

private:
    int64_t total_size_{};
    int64_t shape_[kNdim]{};
};

// Static 0-dimensional specialization.
template <>
class Indexer<0> {
public:
    explicit Indexer(const Shape& shape) {
        CHAINERX_ASSERT(shape.ndim() == 0);
        CHAINERX_ASSERT(shape.GetTotalSize() == 1);
    }

    CHAINERX_HOST_DEVICE IndexIterator<0> It(int64_t start, int64_t step = 1) const { return IndexIterator<0>{start, step}; }

    CHAINERX_HOST_DEVICE static constexpr int8_t ndim() { return 0; }

    CHAINERX_HOST_DEVICE static constexpr int64_t total_size() { return 1; }

    CHAINERX_HOST_DEVICE static constexpr const int64_t* shape() { return nullptr; }
};

// Static 1-dimensional specialization.
template <>
class Indexer<1> {
public:
    explicit Indexer(const Shape& shape) : total_size_{shape[0]} { CHAINERX_ASSERT(1 == shape.ndim()); }

    CHAINERX_HOST_DEVICE IndexIterator<1> It(int64_t start, int64_t step = 1) const { return IndexIterator<1>{total_size_, start, step}; }

    CHAINERX_HOST_DEVICE static constexpr int8_t ndim() { return 1; }

    CHAINERX_HOST_DEVICE int64_t total_size() const { return total_size_; }

    CHAINERX_HOST_DEVICE const int64_t* shape() const { return &total_size_; }

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

    CHAINERX_HOST_DEVICE IndexIterator<kDynamicNdim> It(int64_t start, int64_t step = 1) const {
        return IndexIterator<kDynamicNdim>{shape_, ndim_, total_size_, start, step};
    }

    CHAINERX_HOST_DEVICE int8_t ndim() const { return ndim_; }

    CHAINERX_HOST_DEVICE int64_t total_size() const { return total_size_; }

    CHAINERX_HOST_DEVICE const int64_t* shape() const { return shape_; }

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

}  // namespace chainerx
