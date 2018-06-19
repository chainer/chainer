#pragma once

namespace xchainer {

enum class CopyKind {
    kCopy = 1,
    kView,
};

enum class AveragePoolMode {
    kZero = 1,
    kIgnore,
};

}  // namespace xchainer
