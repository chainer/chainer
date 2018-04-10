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

    // Sets an index from mutiple indexers each of which composes a portion of dimensions in order.
    // indexers must be in the reversed order with regard to dimensions.
    template <typename... Args>
    XCHAINER_HOST_DEVICE void SetIndexers(Args... indexers) {
        int8_t processed_dims = SetIndexersImpl(0, indexers...);
        assert(processed_dims == ndim_);
        assert(std::all_of(shape_, shape_ + ndim_, [ this, i = int8_t{0} ](int64_t) mutable {
            bool ret = 0 <= index_[i] && index_[i] < shape_[i];
            ++i;
            return ret;
        }));
    }

private:
    // Implementation of SetIndexers.
    // Returns the number of written dimensions, which is equal to ndim_.
    // `processed_dim` is the number of written dimensions so far.
    template <typename... Args>
    XCHAINER_HOST_DEVICE int8_t SetIndexersImpl(int8_t processed_dims, const Indexer& last_indexer, Args... indexers) {
        processed_dims = SetIndexersImpl(processed_dims, last_indexer);
        int8_t dims = SetIndexersImpl(processed_dims, indexers...);
        assert(dims == ndim_);
        return dims;
    }

    XCHAINER_HOST_DEVICE int8_t SetIndexersImpl(int8_t processed_dims, const Indexer& indexer) {
        assert(processed_dims + indexer.ndim_ <= ndim_);
        std::copy(indexer.index_, indexer.index_ + indexer.ndim_, &index_[ndim_ - processed_dims - indexer.ndim_]);
        return processed_dims + indexer.ndim_;
    }

    int64_t shape_[kMaxNdim];
    int64_t index_[kMaxNdim];
    int64_t raw_index_;
    int64_t total_size_;
    int8_t ndim_;
};

inline std::ostream& operator<<(std::ostream& os, const Indexer& indexer) {
    std::vector<int64_t> index_vec(indexer.index(), indexer.index() + indexer.ndim());
    Shape shape{indexer.shape(), indexer.shape() + indexer.ndim()};
    Shape index{indexer.index(), indexer.index() + indexer.ndim()};
    return os << "Indexer(shape=" << shape << " index=" << index << ")";
}

}  // namespace xchainer
