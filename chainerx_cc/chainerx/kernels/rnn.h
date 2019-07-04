#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/kernel.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

class RnnGradState {
public:
    RnnGradState() = default;

    virtual ~RnnGradState() = default;

    RnnGradState(const RnnGradState&) = default;
    RnnGradState(RnnGradState&&) = default;
    RnnGradState& operator=(const RnnGradState&) = default;
    RnnGradState& operator=(RnnGradState&&) = default;
};

class RnnKernel : public Kernel {
public:
    static const char* name() { return "Rnn"; }

    virtual std::tuple<std::vector<std::vector<Array>>, std::unique_ptr<RnnGradState>> Call(
            int64_t n_layers,
            Array hx,
            nonstd::optional<Array> cx,
            const std::vector<std::vector<Array>>& ws,
            const std::vector<std::vector<Array>>& bs,
            const std::vector<Array>& xs,
            const int8_t bidirectional,
            const int8_t mode) = 0;
};

class RnnBackwardKernel : public Kernel {
public:
    static const char* name() { return "RnnBackward"; }
    virtual std::vector<std::vector<Array>> Call(
            int64_t n_layers,
            Array hx,
            nonstd::optional<Array> cx,
            const std::vector<std::vector<Array>>& ws,
            const std::vector<std::vector<Array>>& bs,
            const std::vector<Array>& xs,
            Array dhy,
            nonstd::optional<Array> dcy,
            std::vector<Array> ys,
            std::vector<Array> dys,
            const int8_t bidirectional,
            const std::shared_ptr<chainerx::RnnGradState>& state) = 0;
};

}  // namespace chainerx
