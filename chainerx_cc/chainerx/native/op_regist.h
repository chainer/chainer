#pragma once

#include "chainerx/native/native_backend.h"
#include "chainerx/op_registry.h"

// Register an op statically in NativeBackend.
#define CHAINERX_REGISTER_OP_NATIVE(key_op_cls, op_cls)                                         \
    static chainerx::internal::OpRegistrar<chainerx::native::NativeBackend, key_op_cls, op_cls> \
            s_native_backend_op_##op_cls{};  // NOLINT(cert-err58-cpp)
