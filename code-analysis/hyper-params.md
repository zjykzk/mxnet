<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [超参数加载](#%E8%B6%85%E5%8F%82%E6%95%B0%E5%8A%A0%E8%BD%BD)
  - [辅助参数(auxiliary params)](#%E8%BE%85%E5%8A%A9%E5%8F%82%E6%95%B0auxiliary-params)
  - [优化参数(optimizer params)](#%E4%BC%98%E5%8C%96%E5%8F%82%E6%95%B0optimizer-params)
      - [预定义更新策略](#%E9%A2%84%E5%AE%9A%E4%B9%89%E6%9B%B4%E6%96%B0%E7%AD%96%E7%95%A5)
      - [自定义更新策略](#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%9B%B4%E6%96%B0%E7%AD%96%E7%95%A5)
      - [预定义更新学习率方式](#%E9%A2%84%E5%AE%9A%E4%B9%89%E6%9B%B4%E6%96%B0%E5%AD%A6%E4%B9%A0%E7%8E%87%E6%96%B9%E5%BC%8F)
      - [自定义更新学习率](#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%9B%B4%E6%96%B0%E5%AD%A6%E4%B9%A0%E7%8E%87)
      - [使用方式](#%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F)
      - [train rcnn超参数加载](#train-rcnn%E8%B6%85%E5%8F%82%E6%95%B0%E5%8A%A0%E8%BD%BD)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

[TOC]

# 超参数加载

## 辅助参数(auxiliary params)

auxiliary params 是symbol的特殊状态，不会被梯度下降算法更新，例如：做*batch normal*的时候，用到的*moving_mean* 和 *moving_variance*。在启动训练之前设计参数，并在启动的时候传入。

```python
# 辅助参数
aux_params = {
  "batch_mean": [1,3],
  "batch_normal_var": [2, 4]
}

# 开始训练
model.fit(aux_params=aux_params
          #other parameters
         )
```

## 优化参数(optimizer params)

参数的更新策略和更新学习率的方式。

#### 预定义更新策略

`SGD` `NAG` `RMSProp` `Adam` `AdaGrad` `AdaDelta` `DCASGD` `SGLD`

#### 自定义更新策略

```python
# 定义
@mx.optimizer.Optimizer.register
class MyOptimizer(mx.optimizer.Optimizer):
    def update(self, index, weight, grad, state):
        '''需要实现的方法'''
        pass

# 使用
optim = mx.optimizer.Optimizer.create_optimizer('MyOptimizer') # MyOptimizer 这个指大小写不敏感
```

#### 预定义更新学习率方式

`FactorScheduler` ： 每隔n次更新，修改学习率

`MultiFactorScheduler` ：每隔n次更新，修改学习率。与 `FactorScheduler` 区别是每隔修改学习率的参数通过参数的形式传入。

#### 自定义更新学习率

```python
# 定义
class MyScheduler(mx.lr_scheduler.LRScheduler):
    def __call__(self, num_update):
        '''需要实现的方法'''
        pass
    
# 使用
sched = MyScheduler()
```

#### 使用方式

```python
# 学习率更新策略
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler([1,2,3],[0.001,0.003,0.0003])
# 更新策略
optimizer = 'sgd'
# 更新参数，包含更新策略和更新学习率策略的参数
optimizer_params = {
    "momentum": 0.9,
    "wd": 0.00005,
    "learning_rate": 0.001,
    "lr_scheduler": lr_scheduler,
    "rescale_grad": 0.002,
    "clip_gradient": 5
}
# 开始训练
model.fit(optimizer=optimizer,optimizer_params=optimizer_params
          #other parameters
         )
```

#### train rcnn超参数加载

1. [代码在这里](https://github.com/zjykzk/mxnet/blob/master/example/rcnn/train_end2end.py#L124-L144)
2. 更新策略是 `SGD`
3. 更新学习率的方式 是 `MultiFactorScheduler`