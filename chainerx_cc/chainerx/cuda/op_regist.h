#pragma once

#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/op_registry.h"

// Register an op statically in CudaBackend.
#define CHAINERX_REGISTER_OP_CUDA(key_op_cls, op_cls) \
    static chainerx::internal::OpRegistrar<chainerx::cuda::CudaBackend, key_op_cls, op_cls> s_cuda_backend_op_##op_cls{};
