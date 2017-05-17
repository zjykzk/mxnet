#!/usr/bin/env bash

LOG=resenet_fpn.log 

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python train_end2end_fpn.py --network resnet_fpn                      \
                                  --gpu 2,3                                 \
                                  >${LOG} 2>&1 &
