#pragma once

#include <string>

#include <absl/types/optional.h>

namespace chainerx {

absl::optional<std::string> GetEnv(const std::string& name);

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

}  // namespace chainerx
