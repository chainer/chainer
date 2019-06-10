#pragma once

#include <string>

#include <nonstd/optional.hpp>

namespace chainerx {

nonstd::optional<std::string> GetEnv(const std::string& name);

void SetEnv(const std::string& name, const std::string& value);

void UnsetEnv(const std::string& name);

}  // namespace chainerx
