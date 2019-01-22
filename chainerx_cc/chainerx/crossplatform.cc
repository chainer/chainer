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
namespace {

// _WIN32 scoped code is untested.
#ifdef _WIN32
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
#endif  // _WIN32

}  // namespace

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

}  // namespace crossplatform
}  // namespace chainerx
