#pragma once

#include <cstdint>
#include <ostream>

#include "chainerx/constant.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

// Formatter to print StackVector containing integral elements as e.g. '[]' or '[1, 2, 3]'.
class DimsFormatter {
public:
    explicit DimsFormatter(const StackVector<int64_t, kMaxNdim>& dims) : dims_{dims} {}

    DimsFormatter(const DimsFormatter&) = delete;
    DimsFormatter(DimsFormatter&&) = delete;
    DimsFormatter operator=(const DimsFormatter&) = delete;
    DimsFormatter operator=(DimsFormatter&&) = delete;

private:
    void Print(std::ostream& os) const;

    friend std::ostream& operator<<(std::ostream& os, const DimsFormatter& formatter);

    const StackVector<int64_t, kMaxNdim>& dims_;
};

std::ostream& operator<<(std::ostream& os, const DimsFormatter& formatter);

}  // namespace chainerx
