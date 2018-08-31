#pragma once

#include <string>
#include <utility>

#include "chainerx/context.h"
#include "chainerx/graph.h"

namespace chainerx {

class BackpropScope {
public:
    explicit BackpropScope(std::string backprop_name, Context& context = GetDefaultContext())
        : backprop_id_{context.MakeBackpropId(std::move(backprop_name))} {}

    BackpropScope(const BackpropScope&) = delete;
    BackpropScope& operator=(const BackpropScope&) = delete;
    BackpropScope& operator=(BackpropScope&&) = delete;
    BackpropScope(BackpropScope&& other) = delete;

    ~BackpropScope() { backprop_id_.context().ReleaseBackpropIdNoExcept(backprop_id_); }

    BackpropId backprop_id() const { return backprop_id_; }

private:
    BackpropId backprop_id_;
};

}  // namespace chainerx
