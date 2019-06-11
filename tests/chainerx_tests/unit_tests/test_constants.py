import numpy

import chainerx


def test_constants():
    assert chainerx.Inf is numpy.Inf
    assert chainerx.Infinity is numpy.Infinity
    assert chainerx.NAN is numpy.NAN
    assert chainerx.NINF is numpy.NINF
    assert chainerx.NZERO is numpy.NZERO
    assert chainerx.NaN is numpy.NaN
    assert chainerx.PINF is numpy.PINF
    assert chainerx.PZERO is numpy.PZERO
    assert chainerx.e is numpy.e
    assert chainerx.euler_gamma is numpy.euler_gamma
    assert chainerx.inf is numpy.inf
    assert chainerx.infty is numpy.infty
    assert chainerx.nan is numpy.nan
    assert chainerx.newaxis is numpy.newaxis
    assert chainerx.pi is numpy.pi
