#include "xchainer/thread_local_state.h"

#include <type_traits>

namespace xchainer {
namespace internal {

internal::InternalThreadLocalState& GetInternalThreadLocalState() {
    thread_local internal::InternalThreadLocalState t_state{};
    return t_state;
}

}  // namespace internal

ThreadLocalState ThreadLocalState::Get() { return ThreadLocalState{internal::GetInternalThreadLocalState()}; }

void ThreadLocalState::Set(const ThreadLocalState& state) { internal::GetInternalThreadLocalState() = state.state_; }

}  // namespace xchainer
