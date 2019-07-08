#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/dims.h"

namespace chainerx {

Array MaxPool(
        const Array& x,
        const Dims& kernel_size,
        const Dims& stride,
        const Dims& pad,
        bool cover_all = true,
        TensorLayout layout = TensorLayout::NCHW);

enum class AveragePoolPadMode {
    kZero = 1,
    kIgnore,
};

Array AveragePool(
        const Array& x,
        const Dims& kernel_size,
        const Dims& stride,
        const Dims& pad,
        AveragePoolPadMode pad_mode = AveragePoolPadMode::kIgnore,
        TensorLayout layout = TensorLayout::NCHW);

}  // namespace chainerx
