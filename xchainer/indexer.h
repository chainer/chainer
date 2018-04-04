#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ostream>
#include <vector>

#include "xchainer/constant.h"
#include "xchainer/macro.h"
#include "xchainer/shape.h"

namespace xchainer {

// Dynamic-length indexer.
template <int8_t kNdim = kDynamicNdim>
class Indexer {
public:
    explicit Indexer(const Shape& shape) : total_size_(shape.GetTotalSize()), ndim_(shape.ndim()) {
        std::copy(shape.begin(), shape.end(), shape_);
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE int64_t total_size() const { return total_size_; }

    XCHAINER_HOST_DEVICE const int64_t* shape() const { return shape_; }

    XCHAINER_HOST_DEVICE const int64_t* index() const { return index_; }

    XCHAINER_HOST_DEVICE int64_t* index() { return index_; }

    XCHAINER_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    XCHAINER_HOST_DEVICE void Set(int64_t i) {
        assert(0 <= i);
        assert(i < total_size_);
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

template <int8_t kNdim>
std::ostream& operator<<(std::ostream& os, const Indexer<kNdim>& indexer) {
    std::vector<int64_t> index_vec(indexer.index(), indexer.index() + indexer.ndim());
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    Shape index{indexer.index(), indexer.index() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << " index=" << index << ")";
}

}  // namespace xchainer
