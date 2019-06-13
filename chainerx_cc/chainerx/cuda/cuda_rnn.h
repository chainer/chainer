#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/dtype.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

class CudaDevice;

namespace cuda_internal {

// All the public operations in this class are guaranteed to be thread safe.
class CudaRnn {
public:
    std::vector<std::vector<Array>> n_step_rnn(
        CudaDevice& device,
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        const int8_t bidirectional,
        const int8_t mode);

    std::vector<std::vector<Array>> n_step_rnn_backward(
        CudaDevice& device,
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
        const int8_t mode);
};

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
