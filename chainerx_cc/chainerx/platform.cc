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

void SetEnv(const std::string& name, const std::string& value) { windows::SetEnv(name, value); }

void UnsetEnv(const std::string& name) { windows::UnsetEnv(name); }

void* DlOpen(const std::string& filename) { return windows::DlOpen(filename); }

void DlClose(void* handle) { windows::DlClose(handle); }

void* DlSym(void* handle, const std::string& name) { return windows::DlSym(handle, name); }

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

void* DlOpen(const std::string& filename) {
    if (void* handle = ::dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL)) {
        return handle;
    }
    throw ChainerxError{"Failed to load shared object '", filename, "': ", ::dlerror()};
}

void DlClose(void* handle) {
    if (0 != ::dlclose(handle)) {
        throw ChainerxError{"Failed to close shared object: ", ::dlerror()};
    }
}

void* DlSym(void* handle, const std::string& name) {
    if (void* symbol = ::dlsym(handle, name.c_str())) {
        return symbol;
    }

    throw ChainerxError{"Failed to get symbol: ", ::dlerror()};
}

#endif  // _WIN32

}  // namespace platform
}  // namespace chainerx
