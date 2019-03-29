#!/usr/bin/env bash
set -eux

export DEBIAN_FRONTEND=noninteractive
apt-get update
pkgs=(
    git
    g++ cmake clang-5.0 clang-tidy-6.0 clang-6.0 clang-format-6.0
    python3-dev
    wget
    bzip2  # To install miniconda
    parallel  # Used in scripts
)
apt-get install -y "${pkgs[@]}"

update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-6.0 1
update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-6.0 1
