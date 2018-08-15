#include <memory>
#include <string>

#include "xchainer/context.h"
#include "xchainer/native/native_backend.h"

namespace {

class Backend0 : public xchainer::native::NativeBackend {
public:
    using NativeBackend::NativeBackend;

    std::string GetName() const override { return "backend0"; }
};

}  // namespace

extern "C" xchainer::Backend* CreateBackend(xchainer::Context& ctx) { return new Backend0{ctx}; }

extern "C" void DestroyBackend(xchainer::Backend* backend) { delete backend; }
