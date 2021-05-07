#! /usr/bin/env bash
source config.sh

docker build . \
    --build-arg CMAKE_VERSION=${CMAKE_VERSION} \
    --build-arg CMAKE_BUILD=${CMAKE_BUILD} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    --build-arg TORCH_VERSION=${TORCH_VERSION} \
    --build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    --build-arg PCDET_ROOT=${PCDET_ROOT} \
    --build-arg NUSC_ROOT=${NUSC_ROOT} \
    --build-arg CADC_ROOT=${CADC_ROOT} \
    --build-arg KITTI_ROOT=${KITTI_ROOT} \
    --build-arg EVAL_ROOT=${EVAL_ROOT} \
    --build-arg LOGDIR=${LOGDIR} \
    -t pcdet-standalone
