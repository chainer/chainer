#!/usr/bin/env bash
set -eux

download_dir="$1"
conda_dir="$2"


mkdir -p "$download_dir"
wget --quiet 'https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh' -O "$download_dir"/miniconda.sh
bash "$download_dir"/miniconda.sh -b -f -p "$conda_dir"

bin_dir="$conda_dir"/bin

"$bin_dir"/conda config --set changeps1 no
"$bin_dir"/conda update -y -q conda
"$bin_dir"/conda create -y -q --name testenv python=3.6 pip

source "$bin_dir"/activate testenv
pip install -U pip
