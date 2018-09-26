#!/usr/bin/env bash
set -eu

# Usage:
#    run-cpplint.sh [options]
#
# Options:
#    --jobs [JOBS]   The number of concurrent jobs. 1 by default.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="$(realpath "$script_dir"/..)"


parallel_jobs=1
cpplint=cpplint

while [ $# -gt 0 ]; do
    o="$1"
    shift
    case "$o" in
        "--jobs")
            parallel_jobs="$1"
            shift
            ;;
        *)
            echo "$0: Unknown option: $o" >&2
            exit 1
    esac
done


find "${root_dir}"/chainerx \( -name '*.cc' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -type f -print0 | xargs -0 -n1 -P"${parallel_jobs}" "${cpplint}"
