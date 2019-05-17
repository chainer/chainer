#pragma once

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/macro.h"

namespace chainerx {

inline constexpr bool IsDebug() { return CHAINERX_DEBUG; }

}  // namespace chainerx
