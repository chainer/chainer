#!/bin/bash
# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .pfnci/script.sh py37".  This script should also be designed to be
# called in a local machine.  If a local machine has no GPUs, this should fall
# back to CPU testing automatically.
#
# Usage: .pfnci/script.sh [target]
# - target is a test target (e.g., "py37").
#
# Environment variables:
# - GPU (default: 0) ... Set a number of GPUs to GPU.  GPU=0 disables GPU
#       testing.
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.
# - XPYTEST ... Set XPYTEST=/path/to/xpytest-linux for testing xpytest.  It will
#       replace xpytest installed inside a Docker image with the given binary.
#       It should be useful to test xpytest.
# - SPREADSHEET_ID ... Set SPREADSHEET_ID (e.g.,
#       "1u5OYiPOL3XRppn73XBSgR-XyDuHKb_4Ilmx1kgJfa-k") to enable xpytest to
#       report to a spreadsheet.

set -eu

cd "$(dirname "${BASH_SOURCE}")"/..

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"

  # Initialization.
  prepare_docker &
  wait

  # Prepare docker args.
  docker_args=(docker run  --rm --volume="$(pwd):/src:ro")
  if [ "${GPU:-0}" != '0' ]; then
    docker_args+=(--env="GPU=${GPU}" --runtime=nvidia)
  fi
  if [ "${XPYTEST:-}" != '' ]; then
    docker_args+=(--volume="${XPYTEST}:/usr/local/bin/xpytest:ro")
  fi
  if [ "${SPREADSHEET_ID:-}" != '' ]; then
    docker_args+=(--env="SPREADSHEET_ID=${SPREADSHEET_ID}")
  fi

  # Run target-specific commands.
  case "${TARGET}" in
    # Unit tests.
    'py37' | 'py27and35' )
      run "${docker_args[@]}" \
          "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${TARGET}" \
          bash /src/.pfnci/run.sh "${TARGET}"
      ;;
    # Unsupported targets.
    * )
      echo "Unsupported target: ${TARGET}" >&2
      exit 1
      ;;
  esac
}

################################################################################
# Utility functions
################################################################################

# run executes a command.  If DRYRUN is enabled, run just prints the command.
run() {
  echo '+' "$@"
  if [ "${DRYRUN:-}" == '' ]; then
    "$@"
  fi
}

# prepare_docker makes docker use tmpfs to speed up.
# CAVEAT: Do not use docker during this is running.
prepare_docker() {
  # Mount tmpfs to docker's root directory to speed up.
  if [ "${CI:-}" != '' ]; then
    run service docker stop
    run mount -t tmpfs -o size=100% tmpfs /var/lib/docker
    run service docker start
  fi
  # Configure docker to pull images from gcr.io.
  run gcloud auth configure-docker
}

################################################################################
# Bootstrap
################################################################################
main "$@"
