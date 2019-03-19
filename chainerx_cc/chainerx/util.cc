#include "chainerx/util.h"

#include <cstdlib>
#include <string>

#include <nonstd/optional.hpp>

#include "chainerx/platform.h"

namespace chainerx {

nonstd::optional<std::string> GetEnv(const std::string& name) {
    if (const char* value = std::getenv(name.c_str())) {
        return std::string{value};
    }
    return nonstd::nullopt;  // No matching environment variable.
}

void SetEnv(const std::string& name, const std::string& value) { platform::SetEnv(name, value); }

void UnsetEnv(const std::string& name) { platform::UnsetEnv(name); }

}  // namespace chainerx
