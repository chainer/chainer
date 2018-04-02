#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/indexer.h"
#include "xchainer/macro.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace indexable_array_detail {

// Adds `const` to To if From is const
template <typename To, typename From>
using WithConstnessOf = std::conditional_t<std::is_const<From>::value, std::add_const_t<To>, std::remove_const_t<To>>;

}  // namespace indexable_array_detail

// Statically typed array data structure that can be passed to device kernels.
//
// TypedArary only contains the pointer to the first element and the strides information. To access elements with
// linear indexes, Indexer is also needed.
template <typename T, int8_t kNdim = kDynamicNdim>
class IndexableArray {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides, int64_t total_bytes, int64_t offset = 0)
        : data_(data), total_bytes_(total_bytes), offset_(offset) {
        assert(strides.ndim() == kNdim);
        std::copy(strides.begin(), strides.end(), strides_);
    }

    explicit IndexableArray(const Array& array)
        : IndexableArray{reinterpret_cast<T*>(array.raw_data()), array.strides(), array.GetAllocatedBytes(), array.offset()} {
        assert(TypeToDtype<T> == array.dtype());
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto char_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<char, T>*>(data_) + offset_;
        for (int8_t dim = 0; dim < kNdim; ++dim) {
            char_ptr += strides_[dim] * index[dim];
        }
        assert(reinterpret_cast<const char*>(data_) <= char_ptr);
        assert(char_ptr < reinterpret_cast<const char*>(data_) + total_bytes_);
        return *reinterpret_cast<T*>(char_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const Indexer<kNdim>& indexer) const { return operator[](indexer.index()); }

    // Permutes the axes.
    //
    // Given axes may be fewer than that held by the array.
    // In that case, the axes in the array will be reduced.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kDynamicNdim>& Permute(const std::vector<int8_t>& axes) {
        assert(axes.size() <= static_cast<size_t>(kNdim));
        int64_t c[kNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < axes.size(); ++i) {
            strides_[i] = c[axes[i]];
        }
        return *this;
    }

private:
    T* data_;
    int64_t total_bytes_;
    int64_t offset_;
    int64_t strides_[kNdim];
};

// IndexableArray with dynamic ndim.
template <typename T>
class IndexableArray<T, kDynamicNdim> {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides, int64_t total_bytes, int64_t offset = 0)
        : data_(data), total_bytes_(total_bytes), offset_(offset), ndim_(strides.ndim()) {
        std::copy(strides.begin(), strides.end(), strides_);
    }

    explicit IndexableArray(const Array& array)
        : IndexableArray{reinterpret_cast<T*>(array.raw_data()), array.strides(), array.GetAllocatedBytes(), array.offset()} {
        assert(TypeToDtype<T> == array.dtype());
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto char_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<char, T>*>(data_) + offset_;
        for (int8_t dim = 0; dim < ndim_; ++dim) {
            char_ptr += strides_[dim] * index[dim];
        }
        assert(reinterpret_cast<const char*>(data_) <= char_ptr);
        assert(char_ptr < reinterpret_cast<const char*>(data_) + total_bytes_);
        return *reinterpret_cast<T*>(char_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const Indexer<kDynamicNdim>& indexer) const { return operator[](indexer.index()); }

    // Permutes the axes.
    //
    // Given axes may be fewer than that held by the array.
    // In that case, the axes in the array will be reduced.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kDynamicNdim>& Permute(const std::vector<int8_t>& axes) {
        assert(axes.size() <= static_cast<size_t>(ndim_));
        int64_t c[kMaxNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < axes.size(); ++i) {
            strides_[i] = c[axes[i]];
        }
        ndim_ = static_cast<int8_t>(axes.size());
        return *this;
    }

private:
    T* data_;
    int64_t total_bytes_;
    int64_t offset_;
    int64_t strides_[kMaxNdim];
    int8_t ndim_;
};

}  // namespace xchainer
