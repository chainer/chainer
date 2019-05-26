#pragma once

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"

namespace chainerx {

std::vector<std::vector<Array>> n_step_lstm(
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs);

std::vector<std::vector<Array>> n_step_bilstm(
    int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs);
}
