#pragma once

#include <cstdint>

#include "chainerx/constant.h"
#include "chainerx/macro.h"

namespace chainerx {

template <int8_t kNdim = kDynamicNdim>
class CudaIndexIterator {
public:
    explicit __device__ CudaIndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, total_size_{total_size}, raw_index_{0}, start_{start}, step_{step}, index_{} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
        if (total_size > 0) {
            Set(start);
        }
    }

    __device__ CudaIndexIterator<kNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    __device__ CudaIndexIterator<kNdim>& operator--() {
        Set(raw_index_ - step_);
        return *this;
    }

    __device__ void Restart() { Set(start_); }

    __device__ void Restart(int64_t start) {
        assert(start >= 0);
        start_ = start;
        if (total_size_ > 0) {
            Set(start_);
        }
    }

    // TODO(sonots): Set raw_index_
    template <typename IndexSource>
    __device__ void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        for (int i = 0; i < index_source.ndim(); ++i) {
            // NOLINT is required to suppress gsl::at usage since it does not work with CUDA.
            index_[i + offset_dim] = index_source.index()[i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }

    __device__ operator bool() const { return raw_index_ < total_size_; }  // NOLINT(google-explicit-constructor)

    __device__ constexpr int8_t ndim() const { return kNdim; }

    __device__ int64_t raw_index() const { return raw_index_; }

    __device__ int64_t* index() { return index_; }

    __device__ const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    __device__ void Set(int64_t i) {
        CHAINERX_ASSERT(total_size_ > 0);
        raw_index_ = i;
        if (total_size_ >= 1LL << 32) {
            // 64-bit division is very slow on GPU
            uint64_t a = static_cast<uint64_t>(i);
            for (int8_t dim = kNdim; --dim > 0;) {
                uint64_t s = static_cast<uint64_t>(shape_[dim]);
                if (s & (s - 1)) {
                    uint64_t t = a / s;
                    index_[dim] = static_cast<int64_t>(a - t * s);
                    a = t;
                } else {  // exp of 2
                    index_[dim] = static_cast<int64_t>(a & (s - 1));
                    a >>= __popcll(s - 1);
                }
            }
            index_[0] = a;
        } else {
            uint32_t a = static_cast<uint32_t>(i);
            for (int8_t dim = kNdim; --dim > 0;) {
                uint32_t s = static_cast<uint32_t>(shape_[dim]);
                if (s & (s - 1)) {
                    uint32_t t = a / s;
                    index_[dim] = static_cast<int64_t>(a - t * s);
                    a = t;
                } else {  // exp of 2
                    index_[dim] = static_cast<int64_t>(a & (s - 1));
                    a >>= __popc(s - 1);
                }
            }
            index_[0] = a;
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
class CudaIndexIterator<0> {
public:
    explicit __device__ CudaIndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : CudaIndexIterator<0>{start, step} {
        (void)shape;  // unused
        CHAINERX_ASSERT(total_size == 1);
    }

    explicit __device__ CudaIndexIterator(int64_t start, int64_t step) : raw_index_{start} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
    }

    __device__ CudaIndexIterator<0>& operator++() {
        ++raw_index_;
        return *this;
    }

    __device__ CudaIndexIterator<0>& operator--() {
        --raw_index_;
        return *this;
    }

    __device__ void Restart() { raw_index_ = 0; }

    __device__ void Restart(int64_t start) {
        assert(start >= 0);
        raw_index_ = start;
    }

    template <typename IndexSource>
    __device__ void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        (void)index_source;  // unused
        (void)offset_dim;  // unused
        assert(index_source.ndim() == 0);
        assert(offset_dim == 0);
    }

    __device__ operator bool() const { return raw_index_ < 1; }  // NOLINT(google-explicit-constructor)

    __device__ static constexpr int8_t ndim() { return 0; }

    __device__ int64_t raw_index() const { return raw_index_; }

    __device__ int64_t* index() { return &raw_index_; }

    __device__ const int64_t* index() const { return &raw_index_; }

    __device__ void Set(int64_t i) { raw_index_ = i; }

private:
    int64_t raw_index_{0};
};

// Static 1-dimensional specialization.
template <>
class CudaIndexIterator<1> {
public:
    explicit __device__ CudaIndexIterator(const int64_t* shape, int64_t total_size, int64_t start, int64_t step)
        : CudaIndexIterator<1>{total_size, start, step} {
        CHAINERX_ASSERT(shape[0] == total_size);
    }

    explicit __device__ CudaIndexIterator(int64_t total_size, int64_t start, int64_t step)
        : total_size_{total_size}, raw_index_{start}, start_{start}, step_{step} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
    }

    __device__ CudaIndexIterator<1>& operator++() {
        raw_index_ += step_;
        return *this;
    }

    __device__ CudaIndexIterator<1>& operator--() {
        raw_index_ -= step_;
        return *this;
    }

    __device__ void Restart() { raw_index_ = start_; }

    __device__ void Restart(int64_t start) {
        assert(start >= 0);
        raw_index_ = start;
    }

    template <typename IndexSource>
    __device__ void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        (void)index_source;  // unused
        (void)offset_dim;  // unused
        assert(index_source.ndim() == 1);
        assert(offset_dim == 0);
        raw_index_ = index_source.index()[0];
    }

    __device__ operator bool() const { return raw_index_ < total_size_; }  // NOLINT(google-explicit-constructor)

    __device__ static constexpr int8_t ndim() { return 1; }

    __device__ int64_t raw_index() const { return raw_index_; }

    __device__ int64_t* index() { return &raw_index_; }

    __device__ const int64_t* index() const { return &raw_index_; }

private:
    int64_t total_size_{};
    int64_t raw_index_{};
    int64_t start_{};
    int64_t step_{};
};

// Runtime determined dynamic dimension specialization.
template <>
class CudaIndexIterator<kDynamicNdim> {
public:
    explicit __device__ CudaIndexIterator(const int64_t* shape, int8_t ndim, int64_t total_size, int64_t start, int64_t step)
        : shape_{shape}, ndim_{ndim}, total_size_{total_size}, raw_index_{0}, start_{start}, step_{step}, index_{} {
        CHAINERX_ASSERT(start >= 0);
        CHAINERX_ASSERT(step > 0);  // backward iteration is not supported in order to omit lower-bound check for performance.
        if (total_size > 0) {
            Set(start);
        }
    }

    __device__ CudaIndexIterator<kDynamicNdim>& operator++() {
        Set(raw_index_ + step_);
        return *this;
    }

    __device__ CudaIndexIterator<kDynamicNdim>& operator--() {
        Set(raw_index_ - step_);
        return *this;
    }

    __device__ void Restart() { Set(start_); }

    __device__ void Restart(int64_t start) {
        assert(start >= 0);
        start_ = start;
        if (total_size_ > 0) {
            Set(start_);
        }
    }

    // TODO(sonots): Set raw_index_
    template <typename IndexSource>
    __device__ void CopyIndex(IndexSource index_source, int8_t offset_dim = 0) {
        for (int i = 0; i < index_source.ndim(); ++i) {
            index_[i + offset_dim] = index_source.index()[i];  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
        }
    }

    __device__ operator bool() const { return raw_index_ < total_size_; }  // NOLINT(google-explicit-constructor)

    __device__ int8_t ndim() const { return ndim_; }

    __device__ int64_t raw_index() const { return raw_index_; }

    __device__ int64_t* index() { return index_; }

    __device__ const int64_t* index() const { return index_; }

private:
    // Set raw_index_ and index_.
    // i may be out of bounds, but raw_index_ and index_ are updated anyway.
    __device__ void Set(int64_t i) {
        CHAINERX_ASSERT(total_size_ > 0);
        raw_index_ = i;
        if (ndim_ == 0) {
            return;
        }
        if (total_size_ >= 1LL << 32) {
            // 64-bit division is very slow on GPU
            uint64_t a = static_cast<uint64_t>(i);
            for (int8_t dim = ndim_; --dim > 0;) {
                uint64_t s = static_cast<uint64_t>(shape_[dim]);
                if (s & (s - 1)) {
                    uint64_t t = a / s;
                    index_[dim] = static_cast<int64_t>(a - t * s);
                    a = t;
                } else {  // exp of 2
                    index_[dim] = static_cast<int64_t>(a & (s - 1));
                    a >>= __popcll(s - 1);
                }
            }
            index_[0] = a;
        } else {
            uint32_t a = static_cast<uint32_t>(i);
            for (int8_t dim = ndim_; --dim > 0;) {
                uint32_t s = static_cast<uint32_t>(shape_[dim]);
                if (s & (s - 1)) {
                    uint32_t t = a / s;
                    index_[dim] = static_cast<int64_t>(a - t * s);
                    a = t;
                } else {  // exp of 2
                    index_[dim] = static_cast<int64_t>(a & (s - 1));
                    a >>= __popc(s - 1);
                }
            }
            index_[0] = a;
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
