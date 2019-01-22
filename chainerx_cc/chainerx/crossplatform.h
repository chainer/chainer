#pragma once

#include <string>

#include <nonstd/optional.hpp>

namespace chainerx {
namespace crossplatform {

nonstd::optional<std::string> GetEnv(const std::string& name);

int SetEnv(const std::string& name, const std::string& value);

int UnsetEnv(const std::string& name);

}  // namespace crossplatform
}  // namespace chainerx
