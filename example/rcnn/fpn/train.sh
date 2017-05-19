#!/usr/bin/env bash

LOG=resenet_fpn.log 

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python train_end2end_fpn.py --network resnet_fpn                      \
                                  --dataset imagenet_loc_2017               \
                                  --image_set train                         \
                                  --root_path /disk2/data/imagenet_loc_2017 \
                                  --dataset_path ILSVRC                     \
                                  --prefix model/fpn_imagenet_loc_2017      \
                                  --pretrained model/resnet-101             \
                                  --pretrained_epoch 0                      \
                                  --gpu 0                                   \
                                  >${LOG} 2>&1 &
