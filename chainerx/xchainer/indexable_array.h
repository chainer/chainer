#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <gsl/gsl>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend_util.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/indexer.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {
namespace indexable_array_detail {

// Adds `const` to To if From is const
template <typename To, typename From>
using WithConstnessOf = std::conditional_t<std::is_const<From>::value, std::add_const_t<To>, std::remove_const_t<To>>;

static inline std::tuple<const uint8_t*, const uint8_t*> GetDataRange(const Array& a) {
    std::tuple<int64_t, int64_t> range = chainerx::GetDataRange(a.shape(), a.strides(), a.item_size());
    int64_t lower = std::get<0>(range);
    int64_t upper = std::get<1>(range);
    const uint8_t* base = internal::GetRawOffsetData<const uint8_t>(a);
    return std::tuple<const uint8_t*, const uint8_t*>{base + lower, base + upper};
}

}  // namespace indexable_array_detail

template <typename T, int8_t kNdim = kDynamicNdim>
class IndexableArray {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_{data} { std::copy(strides.begin(), strides.end(), strides_); }

    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData<T>(array), strides} {
        XCHAINER_ASSERT(TypeToDtype<T> == array.dtype());

#if XCHAINER_DEBUG
        std::tie(first_, last_) = indexable_array_detail::GetDataRange(array);
#endif
    }

    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    XCHAINER_HOST_DEVICE int8_t ndim() const { return kNdim; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto data_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<uint8_t, T>*>(data_);
        for (int8_t dim = 0; dim < kNdim; ++dim) {
            data_ptr += strides_[dim] * index[dim];
        }
#if XCHAINER_DEBUG
        XCHAINER_ASSERT(first_ == nullptr || first_ <= data_ptr);
        XCHAINER_ASSERT(last_ == nullptr || data_ptr <= last_ - sizeof(T));
#endif  // XCHAINER_DEBUG
        return *reinterpret_cast<T*>(data_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const IndexIterator<kNdim>& it) const { return operator[](it.index()); }

    // Permutes the axes.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kNdim>& Permute(const Axes& axes) {
        XCHAINER_ASSERT(axes.size() == static_cast<size_t>(kNdim));
        int64_t c[kNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < kNdim; ++i) {
            strides_[i] = c[axes[i]];
        }
        return *this;
    }

private:
    T* data_;
#if XCHAINER_DEBUG
    const uint8_t* first_{nullptr};
    const uint8_t* last_{nullptr};
#endif  // XCHAINER_DEBUG
    int64_t strides_[kNdim];
};

// Static 0-dimensional specialization.
template <typename T>
class IndexableArray<T, 0> {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_{data} { XCHAINER_ASSERT(0 == strides.ndim()); }

    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData<T>(array), strides} {
        XCHAINER_ASSERT(TypeToDtype<T> == array.dtype());
    }

    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    XCHAINER_HOST_DEVICE constexpr int8_t ndim() const { return 0; }

    XCHAINER_HOST_DEVICE constexpr const int64_t* strides() const { return nullptr; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        XCHAINER_ASSERT(index == nullptr || index[0] == 0);
        return *data_;
    }

    XCHAINER_HOST_DEVICE T& operator[](const IndexIterator<0>& it) const { return operator[](it.index()); }

    IndexableArray<T, 0>& Permute(const Axes& /*axes*/) {
        // NOOP for 1-dimensional array.
        return *this;
    }

private:
    T* data_;
};

// Static 1-dimensional specialization.
template <typename T>
class IndexableArray<T, 1> {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_{data}, stride_{strides[0]} { XCHAINER_ASSERT(1 == strides.ndim()); }

    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData<T>(array), strides} {
        XCHAINER_ASSERT(TypeToDtype<T> == array.dtype());

#if XCHAINER_DEBUG
        std::tie(first_, last_) = indexable_array_detail::GetDataRange(array);
#endif
    }

    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    XCHAINER_HOST_DEVICE constexpr int8_t ndim() const { return 1; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return &stride_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto data_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<uint8_t, T>*>(data_) + stride_ * index[0];
#if XCHAINER_DEBUG
        XCHAINER_ASSERT(first_ == nullptr || first_ <= data_ptr);
        XCHAINER_ASSERT(last_ == nullptr || data_ptr <= last_ - sizeof(T));
#endif  // XCHAINER_DEBUG
        return *reinterpret_cast<T*>(data_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const IndexIterator<1>& it) const { return operator[](it.index()); }

    IndexableArray<T, 1>& Permute(const Axes& /*axes*/) {
        // NOOP for 1-dimensional array.
        return *this;
    }

private:
    T* data_;
#if XCHAINER_DEBUG
    const uint8_t* first_{nullptr};
    const uint8_t* last_{nullptr};
#endif  // XCHAINER_DEBUG
    int64_t stride_{};
};

// Runtime determined dynamic dimension specialization.
template <typename T>
class IndexableArray<T, kDynamicNdim> {
public:
    using ElementType = T;

    IndexableArray(T* data, const Strides& strides) : data_{data}, ndim_{strides.ndim()} {
        std::copy(strides.begin(), strides.end(), strides_);
    }

    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData<T>(array), strides} {
        XCHAINER_ASSERT(TypeToDtype<T> == array.dtype());

#if XCHAINER_DEBUG
        std::tie(first_, last_) = indexable_array_detail::GetDataRange(array);
#endif
    }

    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    XCHAINER_HOST_DEVICE int8_t ndim() const { return ndim_; }

    XCHAINER_HOST_DEVICE const int64_t* strides() const { return strides_; }

    XCHAINER_HOST_DEVICE T* data() const { return data_; }

    XCHAINER_HOST_DEVICE T& operator[](const int64_t* index) const {
        auto data_ptr = reinterpret_cast<indexable_array_detail::WithConstnessOf<uint8_t, T>*>(data_);
        for (int8_t dim = 0; dim < ndim_; ++dim) {
            data_ptr += strides_[dim] * index[dim];
        }
#if XCHAINER_DEBUG
        XCHAINER_ASSERT(first_ == nullptr || first_ <= data_ptr);
        XCHAINER_ASSERT(last_ == nullptr || data_ptr <= last_ - sizeof(T));
#endif  // XCHAINER_DEBUG
        return *reinterpret_cast<T*>(data_ptr);
    }

    XCHAINER_HOST_DEVICE T& operator[](const IndexIterator<kDynamicNdim>& it) const { return operator[](it.index()); }

    // Permutes the axes.
    //
    // Given axes may be fewer than that held by the array.
    // In that case, the axes in the array will be reduced.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kDynamicNdim>& Permute(const Axes& axes) {
        XCHAINER_ASSERT(axes.size() <= static_cast<size_t>(ndim_));
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
#if XCHAINER_DEBUG
    const uint8_t* first_{nullptr};
    const uint8_t* last_{nullptr};
#endif  // XCHAINER_DEBUG
    int64_t strides_[kMaxNdim];
    int8_t ndim_;
};

}  // namespace chainerx
