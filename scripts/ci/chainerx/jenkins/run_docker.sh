#!/usr/bin/env bash
set -eux

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


test $# -eq 1
host_repo_dir="$1"


# Should be checked out here
test -d "$host_repo_dir"

# Test type
if [ -z "$CHAINERX_JENKINS_TEST_TYPE" ]; then
    echo "CHAINERX_JENKINS_TEST_TYPE is not set. Check the build configuration." >&2
    exit 1
fi


# Variables
container_name="chainerx-ci-${BUILD_ID}-${EXECUTOR_NUMBER}"
container_workspace_dir=/workspace
container_repo_dir=/repo
container_work_dir="$container_workspace_dir"/work
container_conda_dir="$container_workspace_dir"/conda


# Temporary docker build context
context_dir="$(mktemp -d)"
cp "$host_repo_dir"/chainerx_cc/scripts/ci/setup-ubuntu.sh "$context_dir"/
cp "$host_repo_dir"/chainerx_cc/scripts/ci/setup-conda.sh "$context_dir"/
sed 's/{{{UID}}}/'"$UID"'/g' "$this_dir"/Dockerfile.template > "$context_dir"/Dockerfile


# Build docker image
docker build \
       -t image1 \
       --build-arg WORKSPACE_DIR="$container_workspace_dir" \
       --build-arg WORK_DIR="$container_work_dir" \
       --build-arg CONDA_DIR="$container_conda_dir" \
       "$context_dir"


# Boot docker and run test commands
test_command=(bash "$container_repo_dir"/chainerx_cc/scripts/ci/jenkins/run.sh)

# Kill the docker container upon receiving signal
cleanup_container() {
    echo "Terminating docker container: $container_name"
    docker kill "$container_name"
}

trap cleanup_container SIGINT SIGTERM

nvidia-docker run \
     --name "$container_name" \
     --user "$UID" \
     --volume "$host_repo_dir":"$container_repo_dir" \
     --rm \
     -e CHAINERX_JENKINS_BRANCH="$ghprbSourceBranch" \
     -e CHAINERX_JENKINS_WORK_DIR="$container_work_dir" \
     -e CHAINERX_JENKINS_REPO_DIR="$container_repo_dir" \
     -e CHAINERX_JENKINS_CONDA_DIR="$container_conda_dir" \
     -e CHAINERX_JENKINS_TEST_TYPE="$CHAINERX_JENKINS_TEST_TYPE" \
     image1 \
     "${test_command[@]}"
