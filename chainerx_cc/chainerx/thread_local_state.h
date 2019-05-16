#pragma once

#include <utility>

#include "chainerx/backprop_mode.h"
#include "chainerx/context.h"
#include "chainerx/device.h"

namespace chainerx {
namespace internal {

struct InternalThreadLocalState {
    Context* default_context;
    Device* default_device;
    internal::BackpropModeStack backprop_mode_stack;
};

InternalThreadLocalState& GetInternalThreadLocalState();

}  // namespace internal

class ThreadLocalState {
public:
    ThreadLocalState() = default;
    explicit ThreadLocalState(internal::InternalThreadLocalState state) : state_{std::move(state)} {}

    ~ThreadLocalState() = default;

    ThreadLocalState(const ThreadLocalState&) = default;
    ThreadLocalState& operator=(const ThreadLocalState&) = default;
    ThreadLocalState(ThreadLocalState&&) = default;
    ThreadLocalState& operator=(ThreadLocalState&&) = default;

    static ThreadLocalState Get();

    static void Set(const ThreadLocalState& state);

private:
    internal::InternalThreadLocalState state_{};
};

}  // namespace chainerx
