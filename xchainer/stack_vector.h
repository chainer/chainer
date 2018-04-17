#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>

namespace xchainer {
namespace stack_vector_detail {

using size_type = size_t;

template <typename BaseIterator, typename BaseContainer>
class iterator {
public:
    using size_type = stack_vector_detail::size_type;

    using difference_type = typename std::iterator_traits<BaseIterator>::difference_type;
    using value_type = typename std::iterator_traits<BaseIterator>::value_type;
    using pointer = typename std::iterator_traits<BaseIterator>::pointer;
    using reference = typename std::iterator_traits<BaseIterator>::reference;
    using iterator_category = typename std::iterator_traits<BaseIterator>::iterator_category;

    iterator() : it_{BaseIterator{}} {}

    iterator(BaseIterator it) : it_{std::move(it)} {}  // NOLINT: implicitly convertible with base iterator

    // iterator to const_iterator conversion
    template <typename OtherBaseIter>
    iterator(const iterator<
             OtherBaseIter,
             std::enable_if_t<std::is_same<OtherBaseIter, typename BaseContainer::iterator>::value, BaseContainer>>& it_other)
        : it_{it_other.base()} {}

    reference operator*() const { return *it_; }

    reference operator->() const { return it_; }

    bool operator==(const iterator& rhs) const { return it_ == rhs.it_; }
    bool operator!=(const iterator& rhs) const { return it_ != rhs.it_; }

    bool operator<(const iterator& rhs) const { return it_ < rhs.it_; }
    bool operator<=(const iterator& rhs) const { return it_ <= rhs.it_; }
    bool operator>(const iterator& rhs) const { return it_ > rhs.it_; }
    bool operator>=(const iterator& rhs) const { return it_ >= rhs.it_; }

    iterator operator++() {
        ++it_;
        return *this;
    }

    iterator operator++(int) { return iterator{it_++}; }

    iterator& operator--() {
        --it_;
        return *this;
    }

    iterator operator--(int) { return iterator{it_--}; }

    iterator operator+(difference_type n) const { return iterator{it_ + n}; }

    iterator operator-(difference_type n) const { return iterator{it_ - n}; }

    difference_type operator-(const iterator& rhs) const { return it_ - rhs.it_; }

    BaseIterator base() const noexcept { return it_; }

private:
    BaseIterator it_;
};

}  // namespace stack_vector_detail

// Fixed-capacity vector-like container whose buffer can be allocated statically on the stack.
template <typename T, stack_vector_detail::size_type N>
class StackVector {
private:
    using BaseContainer = std::array<T, N>;
    using BaseIterator = typename BaseContainer::iterator;

public:
    using value_type = std::enable_if_t<std::is_default_constructible<T>::value, T>;
    using reference = value_type&;
    using const_reference = const T&;
    using iterator = stack_vector_detail::iterator<typename BaseContainer::iterator, BaseContainer>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_iterator = stack_vector_detail::iterator<typename BaseContainer::const_iterator, BaseContainer>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename iterator::difference_type;
    using size_type = typename iterator::size_type;

    StackVector() {}

    template <typename InputIter>
    StackVector(InputIter first, InputIter last) {
        assert(std::distance(first, last) <= static_cast<difference_type>(N));
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

    StackVector(StackVector&& other) : n_{other.n_} { std::move(other.d_.begin(), other.d_.begin() + other.n_, d_.begin()); }

    StackVector& operator=(StackVector&& other) {
        n_ = other.n_;
        std::move(other.d_.begin(), other.d_.begin() + other.n_, d_.begin());
        return *this;
    }

    const_reference operator[](size_type index) const {
        assert(index < n_);
        return d_[index];
    }

    reference operator[](size_type index) {
        assert(index < n_);
        return d_[index];
    }

    bool operator==(const StackVector& rhs) const { return n_ == rhs.n_ && std::equal(d_.cbegin(), d_.cbegin() + n_, rhs.d_.cbegin()); }
    bool operator!=(const StackVector& rhs) const { return n_ != rhs.n_ || !std::equal(d_.cbegin(), d_.cbegin() + n_, rhs.d_.cbegin()); }

    // iterators
    iterator begin() noexcept { return iterator{d_.begin()}; }
    iterator end() noexcept { return iterator{d_.begin() + n_}; }
    const_iterator begin() const noexcept { return cbegin(); }
    const_iterator end() const noexcept { return cend(); }
    const_iterator cbegin() const noexcept { return const_iterator{d_.cbegin()}; }
    const_iterator cend() const noexcept { return const_iterator{d_.cbegin() + n_}; }

    // reverse iterators
    reverse_iterator rbegin() noexcept { return reverse_iterator{iterator{d_.begin() + n_}}; }
    reverse_iterator rend() noexcept { return reverse_iterator{iterator{d_.begin()}}; }
    const_reverse_iterator rbegin() const noexcept { return crbegin(); }
    const_reverse_iterator rend() const noexcept { return crend(); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator{const_iterator{d_.cbegin() + n_}}; }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator{const_iterator{d_.cbegin()}}; }

    constexpr size_type max_size() const noexcept { return N; }

    size_type size() const noexcept { return n_; }

    bool empty() const noexcept { return n_ == 0; }

    void clear() noexcept { resize(0); }

    void resize(size_type count) noexcept {
        assert(count <= N);
        if (n_ < count) {
            // expanding
            for (size_type i = n_; i < count; ++i) {
                d_[i] = T{};  // default-construct new elements
            }
        } else {
            // shrinking
            for (size_type i = count; i < n_; ++i) {
                d_[i] = T{};  // destruct obsolete elements
            }
        }
        n_ = count;
    }

    reference front() {
        assert(n_ > 0);
        return d_[0];
    }

    const_reference front() const {
        assert(n_ > 0);
        return d_[0];
    }

    reference back() {
        assert(n_ > 0);
        return d_[n_ - 1];
    }

    const_reference back() const {
        assert(n_ > 0);
        return d_[n_ - 1];
    }

    template <typename... Args>
    iterator emplace(const_iterator pos, Args&&... args) {
        assert(n_ < N);
        assert(cbegin() <= pos);
        assert(pos <= cend());
        size_type i_pos = pos - cbegin();
        for (size_type i = n_; i > i_pos; --i) {
            d_[i] = std::move(d_[i - 1]);
        }
        d_[i_pos] = T{std::forward<Args>(args)...};
        ++n_;
        return iterator{d_.begin() + i_pos};
    }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        return *emplace(cend(), std::forward<Args>(args)...);
    }

    void push_back(const T& value) { emplace_back(value); }

    void push_back(T&& value) { emplace_back(std::forward<T>(value)); }

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

}  // namespace xchainer
