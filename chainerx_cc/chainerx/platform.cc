#include "chainerx/platform.h"

#ifdef _WIN32
#include "chainerx/platform/windows.h"
#else  // _WIN32
// Windows doesn't support it currently
#include <dlfcn.h>
// NOLINTNEXTLINE(modernize-deprecated-headers): clang-tidy recommends to use cstdlib, but setenv is not included in cstdlib
#include <stdlib.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "chainerx/error.h"
#endif  // _WIN32

namespace chainerx {
namespace platform {

#ifdef _WIN32

using SetEnv = windows::SetEnv;
using UnsetEnv = windows::UnsetEnv;
using DlOpen = windows::DlOpen;
using DlClose = windows::DlClose;

#else  // _WIN32

void SetEnv(const std::string& name, const std::string& value) {
    if (0 != ::setenv(name.c_str(), value.c_str(), 1)) {
        throw ChainerxError{"Failed to set environment variable '", name, "' to '", value, "': ", std::strerror(errno)};
    }
}

void UnsetEnv(const std::string& name) {
    if (0 != ::unsetenv(name.c_str())) {
        throw ChainerxError{"Failed to unset environment variable '", name, "': ", std::strerror(errno)};
    }
}

void* DlOpen(const std::string& filename, int flags) {
    if (void* handle = ::dlopen(filename.c_str(), flags)) {
        return handle;
    }
    throw ChainerxError{"Failed to load shared object '", filename, "': ", ::dlerror()};
}

void DlClose(void* handle) {
    if (0 != ::dlclose(handle)) {
        throw ChainerxError{"Failed to close shared object: ", ::dlerror()};
    }
}

#endif  // _WIN32

}  // namespace platform
}  // namespace chainerx
