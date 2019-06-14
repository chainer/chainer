import numpy

import chainerx


def test_py_types():
    assert chainerx.bool is bool
    assert chainerx.int is int
    assert chainerx.float is float


def test_dtypes():
    assert chainerx.dtype is numpy.dtype
    assert chainerx.bool_ is numpy.bool_
    assert chainerx.int8 is numpy.int8
    assert chainerx.int16 is numpy.int16
    assert chainerx.int32 is numpy.int32
    assert chainerx.int64 is numpy.int64
    assert chainerx.uint8 is numpy.uint8
    assert chainerx.float16 is numpy.float16
    assert chainerx.float32 is numpy.float32
    assert chainerx.float64 is numpy.float64
