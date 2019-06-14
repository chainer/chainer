#pragma once

#include <memory>

#include "chainerx/context.h"

namespace chainerx {
namespace testing {

class ContextSession {
public:
    ContextSession() : context_{std::make_unique<Context>()}, context_scope_{*context_} {}

    Context& context() { return *context_; }

private:
    std::unique_ptr<Context> context_;
    ContextScope context_scope_;
};

}  // namespace testing
}  // namespace chainerx
