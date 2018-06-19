#pragma once

namespace xchainer {

enum class CopyKind {
    kCopy = 1,
    kView,
};

enum class AveragePoolPadMode {
    kZero = 1,
    kIgnore,
};

}  // namespace xchainer
