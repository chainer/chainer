#pragma once

#include <algorithm>

#include <nonstd/optional.hpp>

namespace xchainer {

class Slice {
public:
    Slice(nonstd::optional<int64_t> start, nonstd::optional<int64_t> stop, nonstd::optional<int64_t> step) : start_(start), stop_(stop), step_(step ? *step : 1) {
        if (step_ == 0) {
            throw DimensionError("Step must not be zero.");
        }
    };
    Slice() = default;
    explicit Slice(int64_t stop) : Slice(0, stop, 1) {}
    Slice(int64_t start, int64_t stop) : Slice(start, stop, 1) {}

    const nonstd::optional<int64_t>& start() const { return start_; }

    const nonstd::optional<int64_t>& stop() const { return stop_; }

    int64_t step() const { return step_; }

    int64_t GetStart(int64_t dim) const { return start_.value_or(step_ > 0 ? 0 : dim - 1); }

    int64_t GetStop(int64_t dim) const { return stop_.value_or(step_ > 0 ? dim : -1); }

    // Returns the number of elements after slicing an axis of length dim.
    int64_t GetLength(int64_t dim) const { return std::max((GetStop(dim) - GetStart(dim)) / step_, int64_t{0}); }

private:
    nonstd::optional<int64_t> start_;
    nonstd::optional<int64_t> stop_;
    int64_t step_ = 1;
};

}  // namespace xchainer
