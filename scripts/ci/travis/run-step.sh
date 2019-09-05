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

# Python program to prepend a timestamp to each line.
# ts from moreutils buffers the data, which could lead to "no output for a while" error.
add_timestamp_py='
import sys
import datetime

def print_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write("[{}] ".format(timestamp))

print_timestamp()

while True:
    c = sys.stdin.read(1)
    if c == "":
        break
    sys.stdout.write(c)
    sys.stdout.flush()
    if c == "\n":
        print_timestamp()
'


cmd='
source "'"$this_dir"'"/steps.sh
source "$CHAINER_BASH_ENV"

set -x
step_"'$step'" '"$@"'
set +x
'
# TODO(niboshi): Keep colorization
bash -e -c "$cmd" | python -c "$add_timestamp_py"
