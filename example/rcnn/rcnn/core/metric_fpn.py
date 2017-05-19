from __future__ import print_function
import mxnet as mx
import numpy as np

from rcnn.config import config

def s_plus_i(s, i):
    return s + '_' + str(i)

def append_i(ss, c):
    return [s_plus_i(s, i) for i in range(c) for s in ss]

def get_rpn_names(c):
    pred = append_i(['rpn_cls_prob','rpn_bbox_loss'], c)
    label = append_i(['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight'], c)
    return pred, label


def get_rcnn_names(c):
    pred = append_i(['rcnn_cls_prob', 'rcnn_bbox_loss'], c)
    label = append_i(['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight'], c)
    if config.TRAIN.END2END:
        pred.extend(append_i(['rcnn_label'], c))
        rpn_pred, rpn_label = get_rpn_names(c)
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


def print_ndarray(a):
    s = a.shape
    if len(s) == 1:
        print('-1-1-1-1-1-1-1-1', a, s)
        print('[', end='')
        for i in range(s[0]):
            print(a[i].asnumpy(), ',', end='')
        print(']')
        return

    print('[', end='')
    for i in range(a.shape[0]):
        print_ndarray(a[i])
    print('],')


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self, c=4):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names(c)
        self.num = c

    def update(self, labels, preds):
        for i in range(self.num):
            print('==============', self.name, i)
            pred = preds[self.pred.index(s_plus_i('rpn_cls_prob', i))]
            label = labels[self.label.index(s_plus_i('rpn_label', i))]

            print('==============1111111')
            print('pred', pred.shape, pred, pred.dtype)
            pred_plus_one = pred+mx.ndarray.ones(pred.shape)
            print(pred_plus_one)
            print_ndarray(pred)
            arg_ch = mx.ndarray.argmax_channel(pred)
            print('pred', arch_ch)
            # pred (b, c, p) or (b, c, h, w)
            pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
            print('==============2222222')
            pred_label = pred_label.reshape((pred_label.shape[0], -1))
            print('==============3333333')
            # label (b, p)
            label = label.asnumpy().astype('int32')

            # filter with keep_inds
            keep_inds = np.where(label != -1)
            pred_label = pred_label[keep_inds]
            label = label[keep_inds]

            print('==============44444444444')
            self.sum_metric += np.sum(pred_label.flat == label.flat)
            print('==============555555555555')
            self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, c=4):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(c)
        self.num = c

    def update(self, labels, preds):
        print('==============', self.name)
        return
        for i in range(self.num):
            print('==============', self.name, i)
            pred = preds[self.pred.index(s_plus_i('rcnn_cls_prob', i))]
            if self.e2e:
                label = preds[self.pred.index(s_plus_i('rcnn_label', i))]
            else:
                label = labels[self.label.index(s_plus_i('rcnn_label', ))]

            last_dim = pred.shape[-1]
            pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
            label = label.asnumpy().reshape(-1,).astype('int32')

            self.sum_metric += np.sum(pred_label.flat == label.flat)
            self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, c=4):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names(c)
        self.num = c

    def update(self, labels, preds):
        print('==============', self.name)
        return
        for i in range(self.num):
            print('==============', self.name, i)
            pred = preds[self.pred.index(s_plus_i('rpn_cls_prob', i))]
            label = labels[self.label.index(s_plus_i('rpn_label', i))]

            # label (b, p)
            label = label.asnumpy().astype('int32').reshape((-1))
            # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
            pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
            pred = pred.reshape((label.shape[0], -1))

            # filter with keep_inds
            keep_inds = np.where(label != -1)[0]
            label = label[keep_inds]
            cls = pred[keep_inds, label]

            cls += 1e-14
            cls_loss = -1 * np.log(cls)
            cls_loss = np.sum(cls_loss)
            self.sum_metric += cls_loss
            self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, c=4):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(c)
        self.num = c

    def update(self, labels, preds):
        print('==============', self.name)
        return
        for i in range(self.num):
            print('==============', self.name, i)
            pred = preds[self.pred.index(s_plus_i('rcnn_cls_prob', i))]
            if self.e2e:
                label = preds[self.pred.index(s_plus_i('rcnn_label', i))]
            else:
                label = labels[self.label.index(s_plus_i('rcnn_label', i))]

            last_dim = pred.shape[-1]
            pred = pred.asnumpy().reshape(-1, last_dim)
            label = label.asnumpy().reshape(-1,).astype('int32')
            cls = pred[np.arange(label.shape[0]), label]

            cls += 1e-14
            cls_loss = -1 * np.log(cls)
            cls_loss = np.sum(cls_loss)
            self.sum_metric += cls_loss
            self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, c=4):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names(c)
        self.num = c

    def update(self, labels, preds):
        print('==============', self.name)
        return
        for i in range(self.num):
            print('==============', self.name, i)
            bbox_loss = preds[self.pred.index(s_plus_i('rpn_bbox_loss', i))].asnumpy()
            bbox_weight = labels[self.label.index(s_plus_i('rpn_bbox_weight', i))].asnumpy()

            # calculate num_inst (average on those fg anchors)
            num_inst = np.sum(bbox_weight > 0) / 4

            self.sum_metric += np.sum(bbox_loss)
            self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, c=4):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names(c)
        self.num = c

    def update(self, labels, preds):
        print('==============', self.name)
        return
        for i in range(self.num):
            print('==============', self.name, i)
            bbox_loss = preds[self.pred.index(s_plus_i('rcnn_bbox_loss', i))].asnumpy()
            if self.e2e:
                label = preds[self.pred.index(s_plus_i('rcnn_label', i))].asnumpy()
            else:
                label = labels[self.label.index(s_plus_i('rcnn_label', i))].asnumpy()

            # calculate num_inst
            keep_inds = np.where(label != 0)[0]
            num_inst = len(keep_inds)

            self.sum_metric += np.sum(bbox_loss)
            self.num_inst += num_inst
