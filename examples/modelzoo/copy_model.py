#!/usr/bin/env python
# coding:utf-8

import chainer

def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
            if isinstance(child, chainer.Link):
                match = True
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    if a[0] != b[0]:
                        match = False
                        break
                    if a[1].data.shape != b[1].data.shape:
                        match = False
                        break
                    if not match:
                        print('Ignore %s because of parameter mismatch' % child.name)
                        continue
                    for a, b in zip(child.namedparams(), dst_child.namedparams()):
                        b[1].data = a[1].data
                        print('Copy %s' % child.name)
