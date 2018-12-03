#!/usr/bin/env bash
set -eu

# Usage:
#    run-clang-format.sh [options]
#
# Options:
#    --jobs [JOBS]   The number of concurrent jobs. 1 by default.
#    --in-place      Apply edits to files in place.

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="$(realpath "$script_dir"/..)"


clang_format=clang-format

parallel_jobs=1
inplace=0

while [ $# -gt 0 ]; do
    o="$1"
    shift
    case "$o" in
        "--in-place")
            inplace=1
            ;;
        "--jobs")
            parallel_jobs="$1"
            shift
            ;;
        *)
            echo "$0: Unknown option: $o" >&2
            exit 1
    esac
done

# {} in cmd is replaced by xargs
if [ "${inplace}" != 0 ]; then
    cmd=("${clang_format}" -i '{}')
else
    cmd=(bash -c 'diff -u {} <('"${clang_format}"' {})')
fi

find "${root_dir}"/chainerx \( -name '*.cc' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) -type f -print0 | xargs -0 -n1 -P"${parallel_jobs}" -I{} "${cmd[@]}"
