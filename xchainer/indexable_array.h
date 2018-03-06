#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include <gsl/gsl>

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
template <typename T, int8_t n_dim = kDynamicNdim>
class IndexableArray {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_(data) {
        Expects(strides.ndim() == n_dim);
        std::copy(strides.begin(), strides.end(), strides_);
    }

    explicit IndexableArray(indexable_array_detail::WithConstnessOf<Array, T>& array)
        : IndexableArray{static_cast<T*>(array.raw_data()), array.strides()} {
        Expects(TypeToDtype<T> == array.dtype());
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return n_dim; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto char_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<char, T>*>(data_);
        for (int8_t dim = 0; dim < n_dim; ++dim) {
            char_ptr += strides_[dim] * index[dim];
        }
        return *reinterpret_cast<T*>(char_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const Indexer<n_dim>& indexer) const { return operator[](indexer.index()); }

private:
    T* data_;
    int64_t strides_[n_dim];
};

// IndexableArray with dynamic ndim.
template <typename T>
class IndexableArray<T, kDynamicNdim> {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_(data), ndim_(strides.ndim()) {
        std::copy(strides.begin(), strides.end(), strides_);
    }

    explicit IndexableArray(indexable_array_detail::WithConstnessOf<Array, T>& array)
        : IndexableArray{static_cast<T*>(array.raw_data()), array.strides()} {
        Expects(TypeToDtype<T> == array.dtype());
    }

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto char_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<char, T>*>(data_);
        for (int8_t dim = 0; dim < ndim_; ++dim) {
            char_ptr += strides_[dim] * index[dim];
        }
        return *reinterpret_cast<T*>(char_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const Indexer<kDynamicNdim>& indexer) const { return operator[](indexer.index()); }

private:
    T* data_;
    int64_t strides_[kMaxNdim];
    int8_t ndim_;
};

}  // namespace xchainer
