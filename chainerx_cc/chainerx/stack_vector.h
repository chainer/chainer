#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include "chainerx/macro.h"

namespace chainerx {
namespace stack_vector_detail {

using size_type = size_t;

}  // namespace stack_vector_detail

// Fixed-capacity vector-like container whose buffer can be allocated statically on the stack.
// Not all features in std::vector are implemented.
template <typename T, stack_vector_detail::size_type N>
class StackVector {
private:
    static_assert(std::is_default_constructible<T>::value, "StackVector requires default constructible element type.");
    static_assert(std::is_trivially_destructible<T>::value, "StackVector requires trivially destructible element type.");
    using BaseContainer = std::array<T, N>;
    using BaseIterator = typename BaseContainer::iterator;

public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename BaseContainer::iterator;
    using reverse_iterator = typename BaseContainer::reverse_iterator;
    using const_iterator = typename BaseContainer::const_iterator;
    using const_reverse_iterator = typename BaseContainer::const_reverse_iterator;
    using difference_type = typename BaseContainer::difference_type;
    using size_type = typename BaseContainer::size_type;

    StackVector() = default;

    ~StackVector() = default;

    template <typename InputIter>
    StackVector(InputIter first, InputIter last) {
        CHAINERX_ASSERT(std::distance(first, last) <= static_cast<difference_type>(N));
        BaseIterator end = std::copy(first, last, d_.begin());
        n_ = std::distance(d_.begin(), end);
    }

    StackVector(std::initializer_list<T> list) : StackVector{list.begin(), list.end()} {}

    StackVector(const StackVector& other) : n_{other.n_} { std::copy(other.d_.cbegin(), other.d_.cbegin() + other.n_, d_.begin()); }

    StackVector& operator=(const StackVector& other) {
        n_ = other.n_;
        std::copy(other.d_.cbegin(), other.d_.cbegin() + other.n_, d_.begin());
        return *this;
    }

    StackVector(StackVector&& other) noexcept : n_{other.n_} { std::move(other.d_.begin(), other.d_.begin() + other.n_, d_.begin()); }

    StackVector& operator=(StackVector&& other) noexcept {
        n_ = other.n_;
        std::move(other.d_.begin(), other.d_.begin() + other.n_, d_.begin());
        return *this;
    }

    const_reference operator[](size_type index) const {
        CHAINERX_ASSERT(index < n_);
        return d_[index];
    }

    reference operator[](size_type index) {
        CHAINERX_ASSERT(index < n_);
        return d_[index];
    }

    bool operator==(const StackVector& rhs) const { return n_ == rhs.n_ && std::equal(d_.cbegin(), d_.cbegin() + n_, rhs.d_.cbegin()); }
    bool operator!=(const StackVector& rhs) const { return n_ != rhs.n_ || !std::equal(d_.cbegin(), d_.cbegin() + n_, rhs.d_.cbegin()); }

    // iterators
    iterator begin() noexcept { return d_.begin(); }
    iterator end() noexcept { return d_.begin() + n_; }
    const_iterator begin() const noexcept { return cbegin(); }
    const_iterator end() const noexcept { return cend(); }
    const_iterator cbegin() const noexcept { return d_.cbegin(); }
    const_iterator cend() const noexcept { return d_.cbegin() + n_; }

    // reverse iterators
    reverse_iterator rbegin() noexcept { return d_.rbegin() + (N - n_); }
    reverse_iterator rend() noexcept { return d_.rbegin() + N; }
    const_reverse_iterator rbegin() const noexcept { return crbegin(); }
    const_reverse_iterator rend() const noexcept { return crend(); }
    const_reverse_iterator crbegin() const noexcept { return d_.crbegin() + (N - n_); }
    const_reverse_iterator crend() const noexcept { return d_.crbegin() + N; }

    constexpr size_type max_size() const noexcept { return N; }

    size_type size() const noexcept { return n_; }

    value_type* data() noexcept { return d_.data(); }

    const value_type* data() const noexcept { return d_.data(); }

    bool empty() const noexcept { return n_ == 0; }

    void clear() noexcept { resize(0); }

    void resize(size_type count) noexcept {
        CHAINERX_ASSERT(count <= N);
        if (n_ < count) {
            // expanding
            for (size_type i = n_; i < count; ++i) {
                d_[i] = T{};  // default-construct new elements
            }
        } else {
            // shrinking: Do nothing because T is required to be trivially destructible.
        }
        n_ = count;
    }

    reference front() {
        CHAINERX_ASSERT(n_ > 0);
        return d_[0];
    }

    const_reference front() const {
        CHAINERX_ASSERT(n_ > 0);
        return d_[0];
    }

    reference back() {
        CHAINERX_ASSERT(n_ > 0);
        return d_[n_ - 1];
    }

    const_reference back() const {
        CHAINERX_ASSERT(n_ > 0);
        return d_[n_ - 1];
    }

    template <typename... Args>
    iterator emplace(const_iterator pos, Args&&... args) {
        CHAINERX_ASSERT(n_ < N);
        CHAINERX_ASSERT(cbegin() <= pos);
        CHAINERX_ASSERT(pos <= cend());
        size_type i_pos = pos - cbegin();
        for (size_type i = n_; i > i_pos; --i) {
            d_[i] = std::move(d_[i - 1]);
        }
        // Using ()-initialization to evade implicit integral promotion.
        // (e.g. int8_t + int8_t = int)
        d_[i_pos] = T(std::forward<Args>(args)...);
        ++n_;
        return iterator{d_.begin() + i_pos};
    }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        return *emplace(cend(), std::forward<Args>(args)...);
    }

    void push_back(const T& value) { emplace_back(value); }

    void push_back(T&& value) { emplace_back(std::forward<T>(value)); }

    iterator insert(const_iterator pos, const_reference value) { return emplace(pos, value); }

    template <class InputIter>
    iterator insert(const_iterator pos, InputIter first, InputIter last) {
        size_type n_old = n_;
        iterator it_pos0 = begin() + std::distance(cbegin(), pos);
        iterator it_pos = it_pos0;
        for (InputIter it = first; it != last; ++it, ++it_pos) {
            emplace(it_pos, *it);
        }
        CHAINERX_ASSERT(n_ == n_old + std::distance(first, last));
        return it_pos0;
    }

    iterator erase(const_iterator pos) { return erase(pos, pos + 1); }

    iterator erase(const_iterator first, const_iterator last) {
        difference_type i_first = std::distance(cbegin(), first);
        BaseIterator it_dst = d_.begin() + i_first;
        for (const_iterator it = last; it != end(); ++it, ++it_dst) {
            *it_dst = std::move(*it);
        }
        n_ -= std::distance(first, last);
        return begin() + i_first;
    }

private:
    size_type n_{0};
    BaseContainer d_{};
};

}  // namespace chainerx
