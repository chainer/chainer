#include "chainerx/thread_local_state.h"

namespace chainerx {
namespace internal {

internal::InternalThreadLocalState& GetInternalThreadLocalState() {
    thread_local internal::InternalThreadLocalState t_state{};
    return t_state;
}

}  // namespace internal

ThreadLocalState ThreadLocalState::Get() { return ThreadLocalState{internal::GetInternalThreadLocalState()}; }

void ThreadLocalState::Set(const ThreadLocalState& state) { internal::GetInternalThreadLocalState() = state.state_; }

}  // namespace chainerx
