#!/usr/bin/env bash
# Runs a single step
set -eu


step=$1
shift

echo "=== Step: $step $@"

export CHAINERX_CI_BASH_ENV=${CHAINERX_CI_BASH_ENV:-"$WORK_DIR"/bash_env}
mkdir -p "$(dirname "$CHAINERX_CI_BASH_ENV")"
touch "$CHAINERX_CI_BASH_ENV"

cmd='
source "'"$REPO_DIR"'"/chainerx/scripts/ci/steps.sh
source "$CHAINERX_CI_BASH_ENV"

set -x
step_"'$step'" '"$@"'
set +x
'
bash -e -c "$cmd"
