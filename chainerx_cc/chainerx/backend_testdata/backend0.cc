#include <memory>
#include <string>

#include "chainerx/context.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/op_registry.h"

namespace {

class Backend0 : public chainerx::native::NativeBackend {
public:
    using NativeBackend::NativeBackend;

    std::string GetName() const override { return "backend0"; }

protected:
    chainerx::OpRegistry& GetParentOpRegistry() override {
        static chainerx::OpRegistry op_registry{&chainerx::native::NativeBackend::GetGlobalOpRegistry()};
        return op_registry;
    }
};

}  // namespace

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
extern "C" chainerx::Backend* CreateBackend(chainerx::Context& ctx) { return new Backend0{ctx}; }

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
extern "C" void DestroyBackend(chainerx::Backend* backend) { delete backend; }
