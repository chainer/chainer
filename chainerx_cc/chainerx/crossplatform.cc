#include "chainerx/crossplatform.h"

// NOLINTNEXTLINE(modernize-deprecated-headers): clang-tidy recommends to use cstdlib, but setenv is not included in cstdlib
#include <stdlib.h>

#include <nonstd/optional.hpp>
#include <string>

#include "chainerx/error.h"

namespace chainerx {
namespace crossplatform {
namespace {

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

int SetEnv(const std::string& name, const std::string& value) { return setenv(name.c_str(), value.c_str(), 1); }

int UnsetEnv(const std::string& name) { return unsetenv(name.c_str()); }

}  // namespace crossplatform
}  // namespace chainerx
