import mxnet as mx

data = mx.sym.Variable('data')
rois = mx.sym.Variable('rois')

operator = mx.symbol.ROIAlign(data=data, rois=rois, pooled_size=(2,2),spatial_scale=1.0)
arg_name = operator.list_arguments()  # get the names of the inputs
out_name = operator.list_outputs()    # get the names of the outputs

# infer shape
arg_shape, out_shape, _ = operator.infer_shape(data=(1,1,6,6), rois=(1,5))
{'input' : dict(zip(arg_name, arg_shape)),
 'output' : dict(zip(out_name, out_shape))}

#Bind with Data and Evaluate
feat_map = mx.nd.array([[[[  0.,   1.,   2.,   3.,   4.,   5.],
                          [  6.,   7.,   8.,   9.,  10.,  11.],
                          [ 12.,  13.,  14.,  15.,  16.,  17.],
                          [ 18.,  19.,  20.,  21.,  22.,  23.],
                          [ 24.,  25.,  26.,  27.,  28.,  29.],
                          [ 30.,  31.,  32.,  33.,  34.,  35.],
                          [ 36.,  37.,  38.,  39.,  40.,  41.],
                          [ 42.,  43.,  44.,  45.,  46.,  47.]]]])

rois_ = mx.nd.array([[0,0,0,4,4]])

ex = operator.bind(ctx=mx.cpu(), args={'data' : feat_map, 'rois' : rois_})
ex.forward()
print 'ROIAlign:\ninput feature map = \n%s \ninput rois = \n%s \n number of outputs = %d\nthe first output = \n%s' % (
           feat_map.asnumpy(), rois_.asnumpy(), len(ex.outputs), ex.outputs[0].asnumpy())

operator2 = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(2,2),spatial_scale=1.0)
ex2 = operator2.bind(ctx=mx.cpu(), args={'data' : feat_map, 'rois' : rois_})
ex2.forward()
print 'ROIPooling:\nnumber of outputs = %d\nthe first output = \n%s' % (
           len(ex.outputs), ex.outputs[0].asnumpy())
