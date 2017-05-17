#!/usr/bin/env bash

LOG=loc_train_loc_2017.log

rm -rf ${LOG}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1

nohup python -m rcnn.tools.train_rpn --network resnet                       \
                                  --dataset imagenet_loc_2017               \
                                  --image_set train                         \
                                  --root_path /disk2/data/imagenet_loc_2017 \
                                  --dataset_path ILSVRC                     \
                                  --prefix model/imagenet_loc_2017          \
                                  --begin_epoch 3                           \
                                  --resume				    \
                                  --gpu 0,1,2,3                             \
                                  >${LOG} 2>&1 &
