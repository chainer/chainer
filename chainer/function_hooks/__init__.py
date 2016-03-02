from chainer.function_hooks import timer
from chainer.function_hooks import debug_print
from chainer.function_hooks import memory


PrintHook = debug_print.PrintHook
MemoryHook = memory.MemoryHook
TimerHook = timer.TimerHook
