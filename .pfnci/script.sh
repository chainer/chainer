#!/bin/bash
# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .pfnci/script.sh py37".  If a machine running the script has no GPUs,
# this should fall back to CPU testing automatically.  This script requires that
# a corresponding Docker image is accessible from the machine.
# TODO(imos): Enable external contributors to test this script on their
# machines.  Specifically, locate a Dockerfile generating chainer-ci-prep.*.
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

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"

  # Initialization.
  prepare_docker &
  wait

  # Prepare docker args.
  docker_args=(docker run  --rm)

  # Run target-specific commands.
  case "${TARGET}" in
    # Unit tests.
    'py37' | 'py27and35' )
      docker_args+=(
          --volume="$(cd "$(dirname "${BASH_SOURCE}")/.."; pwd):/src:ro")
      if [ "${GPU:-0}" != '0' ]; then
        docker_args+=(
            --ipc=host --privileged --env="GPU=${GPU}" --runtime=nvidia)
      fi
      if [ "${XPYTEST:-}" != '' ]; then
        docker_args+=(--volume="${XPYTEST}:/usr/local/bin/xpytest:ro")
      fi
      docker_args+=(
          --env="XPYTEST_NUM_THREADS=${XPYTEST_NUM_THREADS:-$(nproc)}")
      if [ "${SPREADSHEET_ID:-}" != '' ]; then
        docker_args+=(--env="SPREADSHEET_ID=${SPREADSHEET_ID}")
      fi
      run "${docker_args[@]}" \
          "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${TARGET}" \
          bash /src/.pfnci/run.sh "${TARGET}"
      ;;
    # Docker builds.
    docker.* )
      # Parse the target as "docker.{target}.{action}".
      local fragments
      IFS=. fragments=(${TARGET})
      local target="${fragments[1]}"
      local action="${fragments[2]}"
      if [ "${action}" != 'push' -a "${action}" != 'test' ]; then
        echo "Unsupported docker target action: ${action}" >&2
        exit 1
      fi
      # This script can be run in CuPy repository to enable CI to build Chainer
      # base images with a specified CuPy version.  This block enables the
      # script can also be run even in Chainer repository.
      # NOTE: This explicitly pulls CuPy repository instead of pulling from
      # docker because of ensuring its CuPy version.
      local cupy_directory="$(pwd)"
      if [ ! -d "${cupy_directory}/cupy" ]; then
        if [ ! -d .pfnci/cupy ]; then
          run git clone https://github.com/cupy/cupy.git .pfnci/cupy
        fi
        cupy_directory=.pfnci/cupy
      fi
      run docker build -t \
          "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${target}" \
          -f "$(dirname "${BASH_SOURCE}")/${target}.Dockerfile" \
          "${cupy_directory}"
      if [ "${action}" == 'push' ]; then
        run docker push "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${target}"
      fi
      ;;
	'chainermn-cuda92' )
      docker_args+=(
          --volume="$(cd "$(dirname "${BASH_SOURCE}")/.."; pwd):/src:ro")
      if [ "${GPU:-0}" != '0' ]; then
        docker_args+=(
            --ipc=host --privileged --env="GPU=${GPU}" --runtime=nvidia)
      fi
      run "${docker_args[@]}" \
          "asia.gcr.io/pfn-public-ci/chainermn-ci-prep-cuda92" \
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
