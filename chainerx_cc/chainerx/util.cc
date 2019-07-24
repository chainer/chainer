#include "chainerx/util.h"

#include <cstdlib>
#include <string>

#include <absl/types/optional.h>

#include "chainerx/platform.h"

namespace chainerx {

absl::optional<std::string> GetEnv(const std::string& name) {
    if (const char* value = std::getenv(name.c_str())) {
        return std::string{value};
    }
    return absl::nullopt;  // No matching environment variable.
}

void SetEnv(const std::string& name, const std::string& value) { platform::SetEnv(name, value); }

void UnsetEnv(const std::string& name) { platform::UnsetEnv(name); }

}  // namespace chainerx
