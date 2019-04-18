#include <memory>
#include <string>

#include <gsl/gsl>

#include "chainerx/context.h"
#include "chainerx/native/native_backend.h"

namespace {

class Backend1 : public chainerx::native::NativeBackend {
public:
    using NativeBackend::NativeBackend;

    std::string GetName() const override { return "backend1"; }

protected:
    chainerx::KernelRegistry& GetParentKernelRegistry() override {
        static gsl::owner<chainerx::KernelRegistry*> kernel_registry =
                new chainerx::KernelRegistry{&chainerx::native::NativeBackend::GetGlobalKernelRegistry()};
        return *kernel_registry;
    }
};

}  // namespace

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
extern "C" chainerx::Backend* CreateBackend(chainerx::Context& ctx) { return new Backend1{ctx}; }

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
extern "C" void DestroyBackend(chainerx::Backend* backend) { delete backend; }
