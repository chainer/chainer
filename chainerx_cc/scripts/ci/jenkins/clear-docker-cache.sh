# This script is to be copy-pasted as the build script of a Jenkins job.
set -eux

docker images \
       --all \
       --filter 'label=chainerx_test_image=1'

docker images \
       --all \
       --filter 'label=chainerx_test_image=1' \
       --format '{{.ID}}' \
    | xargs docker rmi || true

docker images \
       --all \
       --filter 'label=chainerx_test_image=1'
