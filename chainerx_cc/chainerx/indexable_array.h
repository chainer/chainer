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

template <typename To, typename From>
using WithConstnessOf = dtype_detail::WithConstnessOf<To, From>;

static inline std::tuple<const uint8_t*, const uint8_t*> GetDataRange(const Array& a) {
    std::tuple<int64_t, int64_t> range = chainerx::GetDataRange(a.shape(), a.strides(), a.GetItemSize());
    int64_t lower = std::get<0>(range);
    int64_t upper = std::get<1>(range);
    const uint8_t* base = static_cast<const uint8_t*>(internal::GetRawOffsetData(a));
    return std::tuple<const uint8_t*, const uint8_t*>{base + lower, base + upper};
}

}  // namespace indexable_array_detail

template <typename T, int8_t kNdim = kDynamicNdim>
class IndexableArray {
public:
    using ElementType = T;

private:
    template <typename U>
    using WithConstnessOfT = indexable_array_detail::WithConstnessOf<U, T>;
    using VoidType = WithConstnessOfT<void>;
    using DeviceStorageType = TypeToDeviceStorageType<T>;

public:
    // Suppressing error with the following line that `strides_` is not initialized.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    IndexableArray(VoidType* data, const Strides& strides) : data_{data} { std::copy(strides.begin(), strides.end(), strides_); }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData(array), strides} {
        CHAINERX_ASSERT(TypeToDtype<T> == array.dtype());

#if CHAINERX_DEBUG
        std::tie(first_, last_) = indexable_array_detail::GetDataRange(array);
#endif  // CHAINERX_DEBUG
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    CHAINERX_HOST_DEVICE int8_t ndim() const { return kNdim; }

    CHAINERX_HOST_DEVICE const int64_t* strides() const { return strides_; }

    CHAINERX_HOST_DEVICE VoidType* data() const { return data_; }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const int64_t* index) const {
        auto data_ptr = static_cast<WithConstnessOfT<uint8_t>*>(data_);
        for (int8_t dim = 0; dim < kNdim; ++dim) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            data_ptr += strides_[dim] * index[dim];
        }
#if CHAINERX_DEBUG
        CHAINERX_ASSERT(first_ == nullptr || first_ <= data_ptr);
        CHAINERX_ASSERT(last_ == nullptr || data_ptr <= last_ - sizeof(T));
#endif  // CHAINERX_DEBUG
        return *static_cast<DeviceStorageType*>(static_cast<VoidType*>(data_ptr));
    }

    CHAINERX_HOST_DEVICE WithConstnessOfT<DeviceStorageType>& operator[](const IndexIterator<kNdim>& it) const {
        return operator[](it.index());
    }

    // Permutes the axes.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kNdim>& Permute(const Axes& axes) {
        CHAINERX_ASSERT(axes.size() == static_cast<size_t>(kNdim));
        int64_t c[kNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < kNdim; ++i) {
            strides_[i] = c[axes[i]];
        }
        return *this;
    }

private:
    WithConstnessOfT<void>* data_;
#if CHAINERX_DEBUG
    const uint8_t* first_{nullptr};
    const uint8_t* last_{nullptr};
#endif  // CHAINERX_DEBUG
    int64_t strides_[kNdim];
};

// Static 0-dimensional specialization.
template <typename T>
class IndexableArray<T, 0> {
public:
    using ElementType = T;

private:
    template <typename U>
    using WithConstnessOfT = indexable_array_detail::WithConstnessOf<U, T>;
    using VoidType = WithConstnessOfT<void>;
    using DeviceStorageType = TypeToDeviceStorageType<T>;

public:
    IndexableArray(VoidType* data, const Strides& strides) : data_{data} { CHAINERX_ASSERT(0 == strides.ndim()); }

    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData(array), strides} {
        CHAINERX_ASSERT(TypeToDtype<T> == array.dtype());
    }

    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    CHAINERX_HOST_DEVICE constexpr int8_t ndim() const { return 0; }

    CHAINERX_HOST_DEVICE constexpr const int64_t* strides() const { return nullptr; }

    CHAINERX_HOST_DEVICE VoidType* data() const { return data_; }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const int64_t* index) const {
        CHAINERX_ASSERT(index == nullptr || index[0] == 0);
        return *static_cast<WithConstnessOfT<DeviceStorageType>*>(data_);
    }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const IndexIterator<0>& it) const { return operator[](it.index()); }

    IndexableArray<T, 0>& Permute(const Axes& /*axes*/) {
        // NOOP for 1-dimensional array.
        return *this;
    }

private:
    WithConstnessOfT<void>* data_;
};

// Static 1-dimensional specialization.
template <typename T>
class IndexableArray<T, 1> {
public:
    using ElementType = T;

private:
    template <typename U>
    using WithConstnessOfT = indexable_array_detail::WithConstnessOf<U, T>;
    using VoidType = WithConstnessOfT<void>;
    using DeviceStorageType = TypeToDeviceStorageType<T>;

public:
    IndexableArray(VoidType* data, const Strides& strides) : data_{data}, stride_{strides[0]} { CHAINERX_ASSERT(1 == strides.ndim()); }

    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData(array), strides} {
        CHAINERX_ASSERT(TypeToDtype<T> == array.dtype());

#if CHAINERX_DEBUG
        std::tie(first_, last_) = indexable_array_detail::GetDataRange(array);
#endif  // CHAINERX_DEBUG
    }

    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    CHAINERX_HOST_DEVICE constexpr int8_t ndim() const { return 1; }

    CHAINERX_HOST_DEVICE const int64_t* strides() const { return &stride_; }

    CHAINERX_HOST_DEVICE VoidType* data() const { return data_; }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const int64_t* index) const {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto data_ptr = reinterpret_cast<WithConstnessOfT<uint8_t>*>(data_) + stride_ * index[0];
#if CHAINERX_DEBUG
        CHAINERX_ASSERT(first_ == nullptr || first_ <= data_ptr);
        CHAINERX_ASSERT(last_ == nullptr || data_ptr <= last_ - sizeof(T));
#endif  // CHAINERX_DEBUG
        return *static_cast<DeviceStorageType*>(static_cast<VoidType*>(data_ptr));
    }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const IndexIterator<1>& it) const { return operator[](it.index()); }

    IndexableArray<T, 1>& Permute(const Axes& /*axes*/) {
        // NOOP for 1-dimensional array.
        return *this;
    }

private:
    WithConstnessOfT<void>* data_;
#if CHAINERX_DEBUG
    const uint8_t* first_{nullptr};
    const uint8_t* last_{nullptr};
#endif  // CHAINERX_DEBUG
    int64_t stride_{};
};

// Runtime determined dynamic dimension specialization.
template <typename T>
class IndexableArray<T, kDynamicNdim> {
public:
    using ElementType = T;

private:
    template <typename U>
    using WithConstnessOfT = indexable_array_detail::WithConstnessOf<U, T>;
    using VoidType = WithConstnessOfT<void>;
    using DeviceStorageType = TypeToDeviceStorageType<T>;

public:
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    IndexableArray(WithConstnessOfT<void>* data, const Strides& strides) : data_{data}, ndim_{strides.ndim()} {
        std::copy(strides.begin(), strides.end(), strides_);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    IndexableArray(const Array& array, const Strides& strides) : IndexableArray{internal::GetRawOffsetData(array), strides} {
        CHAINERX_ASSERT(TypeToDtype<T> == array.dtype());

#if CHAINERX_DEBUG
        std::tie(first_, last_) = indexable_array_detail::GetDataRange(array);
#endif  // CHAINERX_DEBUG
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    explicit IndexableArray(const Array& array) : IndexableArray{array, array.strides()} {}

    CHAINERX_HOST_DEVICE int8_t ndim() const { return ndim_; }

    CHAINERX_HOST_DEVICE const int64_t* strides() const { return strides_; }

    CHAINERX_HOST_DEVICE VoidType* data() const { return data_; }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const int64_t* index) const {
        auto data_ptr = static_cast<WithConstnessOfT<uint8_t>*>(data_);
        for (int8_t dim = 0; dim < ndim_; ++dim) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            data_ptr += strides_[dim] * index[dim];
        }
#if CHAINERX_DEBUG
        CHAINERX_ASSERT(first_ == nullptr || first_ <= data_ptr);
        CHAINERX_ASSERT(last_ == nullptr || data_ptr <= last_ - sizeof(T));
#endif  // CHAINERX_DEBUG
        return *static_cast<DeviceStorageType*>(static_cast<VoidType*>(data_ptr));
    }

    CHAINERX_HOST_DEVICE DeviceStorageType& operator[](const IndexIterator<kDynamicNdim>& it) const { return operator[](it.index()); }

    // Permutes the axes.
    //
    // Given axes may be fewer than that held by the array.
    // In that case, the axes in the array will be reduced.
    //
    // It is the caller's responsibility to ensure validity of permutation.
    // If the permutation is invalid, the behavior is undefined.
    IndexableArray<T, kDynamicNdim>& Permute(const Axes& axes) {
        CHAINERX_ASSERT(axes.size() <= static_cast<size_t>(ndim_));
        int64_t c[kMaxNdim]{};
        std::copy(std::begin(strides_), std::end(strides_), c);
        for (size_t i = 0; i < axes.size(); ++i) {
            gsl::at(strides_, i) = gsl::at(c, axes[i]);
        }
        ndim_ = static_cast<int8_t>(axes.size());
        return *this;
    }

private:
    WithConstnessOfT<void>* data_;
#if CHAINERX_DEBUG
    const uint8_t* first_{nullptr};
    const uint8_t* last_{nullptr};
#endif  // CHAINERX_DEBUG
    int64_t strides_[kMaxNdim];
    int8_t ndim_;
};

}  // namespace chainerx
