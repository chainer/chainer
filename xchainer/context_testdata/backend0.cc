#include <memory>
#include <string>

#include "xchainer/context.h"
#include "xchainer/native_backend.h"

namespace {

class Backend0 : public xchainer::NativeBackend {
public:
    using NativeBackend::NativeBackend;

    std::string GetName() const override { return "backend0"; }
};

}  // namespace

extern "C" std::unique_ptr<xchainer::Backend> CreateBackend(xchainer::Context& ctx) { return std::make_unique<Backend0>(ctx); }
