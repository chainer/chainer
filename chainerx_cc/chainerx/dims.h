#pragma once

#include <cstdint>
#include <ostream>

#include "chainerx/constant.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

using Dims = StackVector<int64_t, kMaxNdim>;

// Formatter to print Dims containing integral elements as e.g. '[]' or '[1, 2, 3]'.
class DimsFormatter {
public:
    explicit DimsFormatter(const Dims& dims) : dims_{dims} {}

    ~DimsFormatter() = default;

    DimsFormatter(const DimsFormatter&) = delete;
    DimsFormatter(DimsFormatter&&) = delete;
    DimsFormatter& operator=(const DimsFormatter&) = delete;
    DimsFormatter& operator=(DimsFormatter&&) = delete;

private:
    void Print(std::ostream& os) const;

    friend std::ostream& operator<<(std::ostream& os, const DimsFormatter& formatter);

    const Dims& dims_;
};

std::ostream& operator<<(std::ostream& os, const DimsFormatter& formatter);

}  // namespace chainerx
