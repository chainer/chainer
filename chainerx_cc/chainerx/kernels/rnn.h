#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/kernel.h"
#include "chainerx/stack_vector.h"

namespace chainerx {


class RnnKernel : public Kernel {
public:
    static const char* name() { return "Rnn"; }

    virtual std::vector<std::vector<Array>> Call(
            int64_t n_layers,
            Array hx,
            Array cx,
            const std::vector<std::vector<Array>>& ws,
            const std::vector<std::vector<Array>>& bs,
            std::vector<Array>& xs,
            const int8_t bidirectional,
            const int8_t mode) = 0;
};


class RnnBackwardKernel : public Kernel {
public:
    static const char* name() { return "RnnBackward"; }
    virtual std::vector<std::vector<Array>> Call(
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        Array dhy,
        Array dcy,
        std::vector<Array> ys,
        std::vector<Array> dys,
        const int8_t bidirectional,
        const int8_t mode) = 0;
};

}  // namespace chainerx
