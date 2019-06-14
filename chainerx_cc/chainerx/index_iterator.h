#pragma once

#include <cstdint>

#include "chainerx/constant.h"
#include "chainerx/macro.h"

namespace chainerx {

template <int8_t kNdim = kDynamicNdim>
class IndexIterator {
public:
    explicit CHAINERX_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, total_size_{total_size}, raw_index_{0}, start_{start}, step_{step}, index_{} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
        if (total_size > 0) {
            Set(start);
        }
    }

    CHAINERX_HOST_DEVICE IndexIterator<kNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    CHAINERX_HOST_DEVICE IndexIterator<kNdim>& operator--() {
        Set(raw_index_ - step_);
        return *this;
    }

    CHAINERX_HOST_DEVICE void Restart() { Set(start_); }

    CHAINERX_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        start_ = start;
        if (total_size_ > 0) {
            Set(start_);
        }
    }

    // TODO(sonots): Set raw_index_
    template <typename IndexSource>
    CHAINERX_HOST_DEVICE void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        for (int i = 0; i < index_source.ndim(); ++i) {
            // NOLINT is required to suppress gsl::at usage since it does not work with CUDA.
            index_[i + offset_dim] = index_source.index()[i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }

    CHAINERX_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }  // NOLINT(google-explicit-constructor)

    CHAINERX_HOST_DEVICE constexpr int8_t ndim() const { return kNdim; }

    CHAINERX_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    CHAINERX_HOST_DEVICE int64_t* index() { return index_; }

    CHAINERX_HOST_DEVICE const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    CHAINERX_HOST_DEVICE void Set(int64_t i) {
        CHAINERX_ASSERT(total_size_ > 0);
        raw_index_ = i;
        for (int8_t j = kNdim; --j >= 0;) {
            index_[j] = i % shape_[j];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
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
    explicit CHAINERX_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : IndexIterator<0>{start, step} {
        (void)shape;  // unused
        CHAINERX_ASSERT(total_size == 1);
    }

    explicit CHAINERX_HOST_DEVICE IndexIterator(int64_t start, int64_t step) : raw_index_{start} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
    }

    CHAINERX_HOST_DEVICE IndexIterator<0>& operator++() {
        ++raw_index_;
        return *this;
    }

    CHAINERX_HOST_DEVICE IndexIterator<0>& operator--() {
        --raw_index_;
        return *this;
    }

    CHAINERX_HOST_DEVICE void Restart() { raw_index_ = 0; }

    CHAINERX_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        raw_index_ = start;
    }

    template <typename IndexSource>
    CHAINERX_HOST_DEVICE void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        (void)index_source;  // unused
        (void)offset_dim;  // unused
        assert(index_source.ndim() == 0);
        assert(offset_dim == 0);
    }

    CHAINERX_HOST_DEVICE operator bool() const { return raw_index_ < 1; }  // NOLINT(google-explicit-constructor)

    CHAINERX_HOST_DEVICE static constexpr int8_t ndim() { return 0; }

    CHAINERX_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    CHAINERX_HOST_DEVICE int64_t* index() { return &raw_index_; }

    CHAINERX_HOST_DEVICE const int64_t* index() const { return &raw_index_; }

private:
    int64_t raw_index_{0};
};

// Static 1-dimensional specialization.
template <>
class IndexIterator<1> {
public:
    explicit CHAINERX_HOST_DEVICE IndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : IndexIterator<1>{total_size, start, step} {
        CHAINERX_ASSERT(shape[0] == total_size);
    }

    explicit CHAINERX_HOST_DEVICE IndexIterator(int64_t total_size, int64_t start, int64_t step)
        : total_size_{total_size}, raw_index_{start}, start_{start}, step_{step} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
    }

    CHAINERX_HOST_DEVICE IndexIterator<1>& operator++() {
        raw_index_ += step_;
        return *this;
    }

    CHAINERX_HOST_DEVICE IndexIterator<1>& operator--() {
        raw_index_ -= step_;
        return *this;
    }

    CHAINERX_HOST_DEVICE void Restart() { raw_index_ = start_; }

    CHAINERX_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        raw_index_ = start;
    }

    template <typename IndexSource>
    CHAINERX_HOST_DEVICE void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        (void)index_source;  // unused
        (void)offset_dim;  // unused
        assert(index_source.ndim() == 1);
        assert(offset_dim == 0);
        raw_index_ = index_source.index()[0];
    }

    CHAINERX_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }  // NOLINT(google-explicit-constructor)

    CHAINERX_HOST_DEVICE static constexpr int8_t ndim() { return 1; }

    CHAINERX_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    CHAINERX_HOST_DEVICE int64_t* index() { return &raw_index_; }

    CHAINERX_HOST_DEVICE const int64_t* index() const { return &raw_index_; }

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
    explicit CHAINERX_HOST_DEVICE IndexIterator(const int64_t* shape, int8_t ndim, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, ndim_{ndim}, total_size_{total_size}, raw_index_{0}, start_{start}, step_{step}, index_{} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
        if (total_size > 0) {
            Set(start);
        }
    }

    CHAINERX_HOST_DEVICE IndexIterator<kDynamicNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    CHAINERX_HOST_DEVICE IndexIterator<kDynamicNdim>& operator--() {
        Set(raw_index_ - step_);
        return *this;
    }

    CHAINERX_HOST_DEVICE void Restart() { Set(start_); }

    CHAINERX_HOST_DEVICE void Restart(int64_t start) {
        assert(start >= 0);
        start_ = start;
        if (total_size_ > 0) {
            Set(start_);
        }
    }

    // TODO(sonots): Set raw_index_
    template <typename IndexSource>
    CHAINERX_HOST_DEVICE void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        for (int i = 0; i < index_source.ndim(); ++i) {
            index_[i + offset_dim] = index_source.index()[i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }

    CHAINERX_HOST_DEVICE operator bool() const { return raw_index_ < total_size_; }  // NOLINT(google-explicit-constructor)

    CHAINERX_HOST_DEVICE int8_t ndim() const { return ndim_; }

    CHAINERX_HOST_DEVICE int64_t raw_index() const { return raw_index_; }

    CHAINERX_HOST_DEVICE int64_t* index() { return index_; }

    CHAINERX_HOST_DEVICE const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    CHAINERX_HOST_DEVICE void Set(int64_t i) {
        CHAINERX_ASSERT(total_size_ > 0);
        raw_index_ = i;
        for (int8_t j = ndim_; --j >= 0;) {
            index_[j] = i % shape_[j];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
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

}  // namespace chainerx
