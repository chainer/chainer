#include "chainerx/platform/windows.h"

#include <cerrno>
#include <cstring>
#include <string>

#include "chainerx/error.h"

namespace chainerx {
namespace platform {
namespace windows {

void SetEnv(const std::string& name, const std::string& value) {
    errno_t err = ::_putenv_s(name.c_str(), value.c_str());
    if (err != 0) {
        throw ChainerxError{"Failed to set environment variable '", name, "' to '", value, "': ", std::strerror(errno)};
    }
}

void UnsetEnv(const std::string& name) {
    errno_t err = ::_putenv_s(name.c_str(), "");
    if (err != 0) {
        throw ChainerxError{"Failed to unset environment variable '", name, "': ", std::strerror(errno)};
    }
}

void* DlOpen(const std::string& filename) {
    // TODO(hvy): Implement dlopen for Windows.
    throw ChainerxError{"dlopen not implemented for Windows."};
}

void DlClose(void* handle) {
    // TODO(hvy): Implement dlclose for Windows.
    throw ChainerxError{"dlclose not implemented for Windows."};
}

void* DlSym(void* handle, const std::string& name) {
    // TODO(swd): Implement dlsym for Windows.
    throw ChainerxError{"dlsym not implemented for Windows."};
}

}  // namespace windows
}  // namespace platform
}  // namespace chainerx
