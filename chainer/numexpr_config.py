"""Numexpr configuration for Chainer

When available, numexpr (<https://github.com/pydata/numexpr>) can speed up
activation function calculations and backprop on the CPU by an order of
magnitude.
"""

numexpr_enabled = False

try:
    from numexpr import evaluate  # NOQA
    numexpr_enabled = True
    import numexpr
    numexpr = numexpr
except ImportError:
    numexpr = None
