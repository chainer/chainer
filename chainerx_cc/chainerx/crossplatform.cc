#include "chainerx/crossplatform.h"

#ifndef _WIN32
// Windows doesn't support it currently
#include <dlfcn.h>
#endif  // _WIN32
// NOLINTNEXTLINE(modernize-deprecated-headers): clang-tidy recommends to use cstdlib, but setenv is not included in cstdlib
#include <stdlib.h>

#include <cerrno>
#include <cstring>
#include <string>

#include <nonstd/optional.hpp>

#include "chainerx/error.h"

namespace chainerx {
namespace crossplatform {

// _WIN32 scoped code is untested.
#ifdef _WIN32
namespace {

int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite) {
        size_t required_count = 0;
        auto err = getenv_s(&required_count, nullptr, 0, name);
        if (err != 0 || required_count != 0) {
            return err;
        }
    }
    return _putenv_s(name, value);
}

int unsetenv(const char* name) { return _putenv_s(name, ""); }

void* dlopen(const char* /*filename*/, int /*flags*/) {
    // TODO(hvy): Implement dlopen for Windows.
    throw ChainerxError{"dlopen not implemented for Windows."};
}

int dlclose(void* /*handle*/) {
    // TODO(hvy): Implement dlclose for Windows.
    throw ChainerxError{"dlclose not implemented for Windows."};
}

char* dlerror() {
    // TODO(hvy): Implement dlerror for Windows.
    throw ChainerxError{"dlerror not implemented for Windows."};
}

}  // namespace
#endif  // _WIN32

nonstd::optional<std::string> GetEnv(const std::string& name) {
    if (const char* value = getenv(name.c_str())) {
        return std::string{value};
    }
    return nonstd::nullopt;  // No matching environment variable.
}

void SetEnv(const std::string& name, const std::string& value) {
    if (setenv(name.c_str(), value.c_str(), 1)) {
        throw ChainerxError{"Failed to set environment variable ", name, " to ", value, ": ", std::strerror(errno)};
    }
}

void UnsetEnv(const std::string& name) {
    if (unsetenv(name.c_str())) {
        throw ChainerxError{"Failed to unset environment variable ", name, ": ", std::strerror(errno)};
    }
}

void* DlOpen(const std::string& filename, int flags) {
    if (void* handle = dlopen(filename.c_str(), flags)) {
        return handle;
    }
    throw ChainerxError{"Could not load shared object ", filename, ": ", dlerror()};
}

void DlCloseNoExcept(void* handle) noexcept { dlclose(handle); }

}  // namespace crossplatform
}  // namespace chainerx
