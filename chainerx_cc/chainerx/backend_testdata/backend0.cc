#include <memory>
#include <string>

#include "chainerx/context.h"
#include "chainerx/kernel_registry.h"
#include "chainerx/native/native_backend.h"

namespace {

class Backend0 : public chainerx::native::NativeBackend {
public:
    using NativeBackend::NativeBackend;

    std::string GetName() const override { return "backend0"; }

protected:
    chainerx::KernelRegistry& GetParentKernelRegistry() override {
        static chainerx::KernelRegistry kernel_registry{&chainerx::native::NativeBackend::GetGlobalKernelRegistry()};
        return kernel_registry;
    }
};

}  // namespace

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
extern "C" chainerx::Backend* CreateBackend(chainerx::Context& ctx) { return new Backend0{ctx}; }

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
extern "C" void DestroyBackend(chainerx::Backend* backend) { delete backend; }
