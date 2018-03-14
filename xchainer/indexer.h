#pragma once

#include <algorithm>
#include <cstdint>

#include <gsl/gsl>

#include "xchainer/constant.h"
#include "xchainer/macro.h"
#include "xchainer/shape.h"

namespace xchainer {

// Kernel object to index typed arrays.
//
// Indexer holds the shape information. It can be used to access elements of IndexableArray by linear indexes.
template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_(shape.GetTotalSize()) {
        Expects(shape.size() == kNdim);
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE void Set(int64_t index) {
        raw_index_ = index;
        for (int8_t i = kNdim; --i >= 0;) {
            index_[i] = index % shape_[i];
            index /= shape_[i];
        }
    }

private:
    int64_t shape_[kNdim];
    int64_t index_[kNdim];
    int64_t raw_index_;
    int64_t total_size_;
};

// Dynamic-length indexer.
template <>
class Indexer<kDynamicNdim> {
public:
    explicit Indexer(const Shape& shape) : total_size_(shape.GetTotalSize()), ndim_(shape.ndim()) {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        raw_index_ = i;
        for (int8_t j = ndim_; --j >= 0;) {
            index_[j] = i % shape_[j];
            i /= shape_[j];
        }
    }

private:
    int64_t shape_[kMaxNdim];
    int64_t index_[kMaxNdim];
    int64_t raw_index_;
    int64_t total_size_;
    int8_t ndim_;
};

}  // namespace xchainer
