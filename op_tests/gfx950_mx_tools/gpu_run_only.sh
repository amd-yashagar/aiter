#!/bin/bash
# Inside a gpuq lease: aiter env for our clone.
W=/home/yashagar/work/wt/GM-123/mx-work
export TMPDIR=$W/tmp FLYDSL_RUNTIME_CACHE_DIR=$W/flycache AITER_MOE_DISABLE=1
export PYTHONPATH=/home/yashagar/work/wt/GM-123/aiter-mx-dev:$W
cd $W
export FLYDSL_RUNTIME_RUN_ONLY=1
exec "$@"
