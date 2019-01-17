#pragma once
#include <cmath>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace chainerx {

bool AllClose(const Array& a, const Array& b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);

template <typename T>
inline bool IsNan(T /*value*/) {
    return false;
}

inline bool IsNan(float value) { return std::isnan(value); }
inline bool IsNan(double value) { return std::isnan(value); }

template <typename T>
inline bool IsInf(T /*value*/) {
    return false;
}

inline bool IsInf(double value) { return std::isinf(value); }
inline bool IsInf(float value) { return std::isinf(value); }

}  // namespace chainerx
