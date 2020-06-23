#!/bin/bash
# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .pfnci/script.sh py37".  If a machine running the script has no GPUs,
# this should fall back to CPU testing automatically.  This script requires that
# a corresponding Docker image is accessible from the machine.
# TODO(imos): Enable external contributors to test this script on their
# machines.
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

  # Base development branch name.
  base_branch="$(get_base_branch)"
  cupy_branch="${base_branch}"
  if [ "${cupy_branch}" = "master" ]; then
    cupy_branch="v7"
  fi

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
          "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${TARGET}:${base_branch}" \
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
      if [ "${action}" = "push" ] && ! is_known_base_branch "${FLEXCI_BRANCH}"; then
        echo "Branch invalid for docker push: ${FLEXCI_BRANCH}" >&2
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
          run git -C .pfnci/cupy checkout "${cupy_branch}"
        fi
        cupy_directory=.pfnci/cupy
      fi
      run docker build -t \
          "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${target}:${base_branch}" \
          -f "$(dirname "${BASH_SOURCE}")/${target}.Dockerfile" \
          "${cupy_directory}"
      if [ "${action}" == 'push' ]; then
        run docker push "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${target}:${base_branch}"
      fi
      ;;
    'chainermn' )
      docker_args+=(
          --volume="$(cd "$(dirname "${BASH_SOURCE}")/.."; pwd):/src:ro")
      if [ "${GPU:-0}" != '0' ]; then
        docker_args+=(
            --ipc=host --privileged --env="GPU=${GPU}" --runtime=nvidia)
      fi
      run "${docker_args[@]}" \
          "asia.gcr.io/pfn-public-ci/chainer-ci-prep.${TARGET}:${base_branch}" \
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
  echo '+' "$@" >&2
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

# is_known_base_branch returns 0 only if the given branch name is a known
# base development branch.
is_known_base_branch() {
  local branch="${1##refs/heads/}"
  for BASE_BRANCH in master v7; do
    if [ "${branch}" = "${BASE_BRANCH}" ]; then
      return 0
    fi
  done
  return 1
}

# get_base_branch returns the base development branch for the current HEAD.
get_base_branch() {
  for BASE_BRANCH in master v7; do
    run git merge-base --is-ancestor "origin/${BASE_BRANCH}" HEAD && echo "${BASE_BRANCH}" && return 0
  done
  echo "Base branch of HEAD is not valid." >&2
  return 1
}

################################################################################
# Bootstrap
################################################################################
main "$@"
