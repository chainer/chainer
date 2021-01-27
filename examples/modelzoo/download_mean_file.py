#!/usr/bin/env python
import six


print('Downloading ILSVRC12 mean file for NumPy...')
six.moves.urllib.request.urlretrieve(
    'https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/'
    'ilsvrc_2012_mean.npy',
    'ilsvrc_2012_mean.npy')
print('Done')
