#! /usr/bin/env bash
source config.sh

docker run \
    --name="pcdet-standalone.$(whoami).$RANDOM" \
    --gpus=all \
    --rm \
    -it \
    -v "${HOST_PCDET_ROOT}":"${PCDET_ROOT}" \
    -v "${HOST_NUSC_ROOT}":"${NUSC_ROOT}" \
    -v "${HOST_CADC_ROOT}":"${CADC_ROOT}" \
    -v "${HOST_KITTI_ROOT}":"${KITTI_ROOT}" \
    -v "${HOST_EVAL_ROOT}":"${EVAL_ROOT}" \
    -v "${HOST_LOGDIR}":"${LOGDIR}" \
    $@ \
    pcdet-standalone
