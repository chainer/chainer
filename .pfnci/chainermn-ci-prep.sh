#!/bin/bash
# Usage: .pfnci/chainermn-ci-prep.sh [target]
# - target is a test target (e.g., "chainermn-ci-prep-cuda92").
#
# For testing this script, set DRYRUN=1 as an environment variable and run the
# script.  It will print commands without execution.
# $ DRYRUN=1 bash .pfnci/script.sh chainermn-ci-prep-cuda92

set -eu

cd "$(dirname "${BASH_SOURCE}")"/..

main() {
  TARGET="$1"
  BASE_IMAGE="$2"

  prepare_docker &
  wait

  run docker build -t "asia.gcr.io/pfn-public-ci/${TARGET}" \
      --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
      -f ".pfnci/chainermn-ci-prep.Dockerfile" .
  run docker push "asia.gcr.io/pfn-public-ci/${TARGET}"
}

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
  if [ "${CI:-}" != '' ]; then
    run service docker stop
    run mount -t tmpfs -o size=100% tmpfs /var/lib/docker
    run service docker start
  fi
  run gcloud auth configure-docker
}

main "$@"
