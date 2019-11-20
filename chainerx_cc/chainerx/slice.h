#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include <absl/types/optional.h>

#include "chainerx/error.h"

namespace chainerx {

class Slice {
public:
    Slice(absl::optional<int64_t> start, absl::optional<int64_t> stop, absl::optional<int64_t> step)
        : start_{start}, stop_{stop}, step_{step.value_or(1)} {
        if (step_ == 0) {
            throw DimensionError{"Step must not be zero."};
        }
    }
    Slice() = default;
    explicit Slice(int64_t stop) : Slice{0, stop, 1} {}
    Slice(int64_t start, int64_t stop) : Slice{start, stop, 1} {}

    absl::optional<int64_t> start() const { return start_; }

    absl::optional<int64_t> stop() const { return stop_; }

    int64_t step() const { return step_; }

    // For positive `step_`, this function returns 0 to `dim`,
    // inclusive. For negative `step_`, this function returns -1 to
    // `dim - 1`, inclusive.
    int64_t GetStart(int64_t dim) const {
        if (start_.has_value()) {
            int64_t first_valid_start = step_ > 0 ? 0 : -1;
            if (*start_ < 0) {
                return std::max(first_valid_start, *start_ + dim);
            }
            return std::min(*start_, dim + first_valid_start);
        }
        return step_ > 0 ? 0 : dim - 1;
    }

    // Unlike `GetStart`, this function returns -1 to `dim` inclusive
    // not depending on the sign of `step_`. -1 for positive `step_`
    // is equivalent to 0 and `dim` for negative `step_` is equivalent
    // to `dim - 1`, respectively, thanks to the calculation of
    // `GetLength`.
    int64_t GetStop(int64_t dim) const {
        if (stop_.has_value()) {
            if (*stop_ < 0) {
                return std::max(int64_t{-1}, *stop_ + dim);
            }
            return std::min(*stop_, dim);
        }
        return step_ > 0 ? dim : -1;
    }

    // Returns the number of elements after slicing an axis of length dim.
    int64_t GetLength(int64_t dim) const {
        // TODO(hvy): Round according to step sign, nicely.
        return std::max(int64_t{0}, (GetStop(dim) - GetStart(dim) + step_ + (step_ > 0 ? -1 : 1)) / step_);
    }

private:
    absl::optional<int64_t> start_;
    absl::optional<int64_t> stop_;
    int64_t step_{1};
};

}  // namespace chainerx
