#!/usr/bin/env bash
set -eux

export DEBIAN_FRONTEND=noninteractive
apt-get update
pkgs=(
    git
    cmake
    g++-8 clang-3.9
    clang-tidy-5.0 clang-6.0 clang-format-6.0
    python3-dev
    wget
    bzip2  # To install miniconda
    parallel  # Used in scripts
)
apt-get install -y "${pkgs[@]}"

# Clang
apt-get update
apt-get install -y \
    curl \
    zlib1g-dev \

curl https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main" > /etc/apt/sources.list.d/llvm.list
apt-get update

apt-get install -y \
    llvm-3.9 \


# Setup alternatives
update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-5.0 1
update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-6.0 1
