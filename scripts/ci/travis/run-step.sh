#!/usr/bin/env bash
# Runs a single step
set -eu
set -o pipefail

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

test -f "$CHAINER_BASH_ENV"
test -d "$REPO_DIR"


step=$1
shift

echo "=== Step: $step $@"

cmd='
source "'"$this_dir"'"/steps.sh
source "$CHAINER_BASH_ENV"

set -x
step_"'$step'" '"$@"'
set +x
'
# Using ts to add timestamp.
# TODO(niboshi): Keep colorization
bash -e -c "$cmd" | ts '[%Y-%m-%d %H:%M:%S]'
