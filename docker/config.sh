#! /usr/bin/env bash

# Build component versions
CMAKE_VERSION=3.16
CMAKE_BUILD=5
PYTHON_VERSION=3.6
TORCH_VERSION=1.8.1
TORCHVISION_VERSION=0.9.1

# Workspace structure in docker
PCDET_ROOT=/root/pcdet
NUSC_ROOT=/root/nusc
CADC_ROOT=/root/cadc
KITTI_ROOT=/root/kitti
LOGDIR=/root/logdir

# Workspace structure on host machine
HOST_PCDET_ROOT=/home/matthew/git/cadc_testing/WISEPCDet_MIMO
HOST_NUSC_ROOT=/path/to/nuscenes
HOST_CADC_ROOT=/path/to/cadc
HOST_KITTI_ROOT=/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/kitti
HOST_LOGDIR=/home/matthew/git/cadc_testing/pcdet_output
