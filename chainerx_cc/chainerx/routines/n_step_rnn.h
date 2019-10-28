#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"

namespace chainerx {

std::vector<std::vector<Array>> NStepLstm(
        int64_t n_layers,
        const Array& hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs);

std::vector<std::vector<Array>> NStepBiLstm(
        int64_t n_layers,
        const Array& hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs);

std::vector<std::vector<Array>> NStepGru(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs);

std::vector<std::vector<Array>> NStepBiGru(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs);

std::vector<std::vector<Array>> NStepRnn(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        absl::optional<std::string> activation);

std::vector<std::vector<Array>> NStepBiRnn(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        absl::optional<std::string> activation);
}  // namespace chainerx
