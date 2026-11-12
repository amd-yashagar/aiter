#!/bin/bash
# Host-only FlyDSL compile: no GPU visible, nothing launched.
W=/home/yashagar/work/wt/GM-123/mx-work
export AITER_AOT_REPLAY=1 CU_NUM=256 GPU_ARCHS=gfx950 COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx950 ROCR_VISIBLE_DEVICES=NONE HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1
export TMPDIR=$W/tmp FLYDSL_RUNTIME_CACHE_DIR=$W/flycache
export PYTHONPATH=/home/yashagar/work/wt/GM-123/aiter-mx-dev:$W
exec "$@"
