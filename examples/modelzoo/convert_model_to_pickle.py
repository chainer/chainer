#読み込むcaffeモデルとpklファイルを保存するパス
loadpath = "bvlc_alexnet.caffemodel"
savepath = "./chainermodels/alexnet.pkl"

from chainer.links.caffe import CaffeFunction
alexnet = CaffeFunction(loadpath)

import _pickle as pickle
pickle.dump(alexnet, open(savepath, 'wb'))
