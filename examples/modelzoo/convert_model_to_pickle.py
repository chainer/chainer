loadpath = "bvlc_alexnet.caffemodel"
savepath = "alexnet.pkl"

from chainer.links.caffe import CaffeFunction
alexnet = CaffeFunction(loadpath)

import cPickle as pickle
pickle.dump(alexnet, open(savepath, 'wb'))
