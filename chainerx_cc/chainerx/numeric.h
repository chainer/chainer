#pragma once
#include <cmath>

#include "chainerx/array.h"
#include "chainerx/scalar.h"

namespace std {
inline bool isinf(bool value) { return false; }
inline bool isinf(signed char value) { return false; }
inline bool isinf(unsigned char value) { return false; }
inline bool isinf(short value) { return false; }
inline bool isinf(int value) { return false; }
inline bool isinf(unsigned int value) { return false; }
inline bool isinf(long long value) { return false; }

inline bool isnan(bool value) { return false; }
inline bool isnan(signed char value) { return false; }
inline bool isnan(unsigned char value) { return false; }
inline bool isnan(short value) { return false; }
inline bool isnan(unsigned short value) { return false; }
inline bool isnan(int value) { return false; }
inline bool isnan(unsigned int value) { return false; }
inline bool isnan(long long value) { return false; }
inline bool isnan(unsigned long long value) { return false; }
}  // namespace std

namespace chainerx {

bool AllClose(const Array& a, const Array& b, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false);

}  // namespace chainerx
