import datetime
import os
import sys
import time
if os.name == 'nt':
    import ctypes

    _STD_OUTPUT_HANDLE = -11

    class _COORD(ctypes.Structure):
        _fields_ = [('X', ctypes.c_short), ('Y', ctypes.c_short)]

    class _SMALL_RECT(ctypes.Structure):
        _fields_ = [('Left', ctypes.c_short), ('Top', ctypes.c_short),
                    ('Right', ctypes.c_short), ('Bottom', ctypes.c_short)]

    class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
        _fields_ = [('dwSize', _COORD), ('dwCursorPosition', _COORD),
                    ('wAttributes', ctypes.c_ushort),
                    ('srWindow', _SMALL_RECT),
                    ('dwMaximumWindowSize', _COORD)]

    def set_console_cursor_position(x, y):
        """Set relative cursor position from current position to (x,y)"""

        whnd = ctypes.windll.kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        ctypes.windll.kernel32.GetConsoleScreenBufferInfo(whnd,
                                                          ctypes.byref(csbi))
        cur_pos = csbi.dwCursorPosition
        pos = _COORD(cur_pos.X + x, cur_pos.Y + y)
        ctypes.windll.kernel32.SetConsoleCursorPosition(whnd, pos)

    def erase_console(x, y, mode=0):
        """Erase screen.

        Mode=0: From (x,y) position down to the bottom of the screen.
        Mode=1: From (x,y) position down to the begining of line.
        Mode=2: Hole screen
        """

        whnd = ctypes.windll.kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        csbi = _CONSOLE_SCREEN_BUFFER_INFO()
        ctypes.windll.kernel32.GetConsoleScreenBufferInfo(whnd,
                                                          ctypes.byref(csbi))
        cur_pos = csbi.dwCursorPosition
        wr = ctypes.c_ulong()
        if mode == 0:
            num = csbi.srWindow.Right * (csbi.srWindow.Bottom -
                                         cur_pos.Y) - cur_pos.X
            ctypes.windll.kernel32.FillConsoleOutputCharacterA(
                whnd, ord(' '), num, cur_pos, ctypes.byref(wr))
        elif mode == 1:
            num = cur_pos.X
            ctypes.windll.kernel32.FillConsoleOutputCharacterA(
                whnd, ord(' '), num, _COORD(0, cur_pos.Y), ctypes.byref(wr))
        elif mode == 2:
            os.system('cls')


class IteratorProgressBar(object):

    def __init__(self, iterator, title=None, bar_length=50, out=sys.stdout):
        self._iterator = iterator
        self._title = '' if title is None else title
        self._bar_length = bar_length
        self._out = out
        self._recent_timing = []

    def update(self):
        it = self._iterator
        title = self._title
        bar_length = self._bar_length
        out = self._out
        recent_timing = self._recent_timing

        if os.name == 'nt':
            erase_console(0, 0)
        else:
            out.write('\033[J')

        now = time.time()
        recent_timing.append(
            (it.current_position, it.epoch_detail, now))

        old_t, old_e, old_sec = recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (it.current_position - old_t) / span
            speed_e = (it.epoch_detail - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')

        estimated_time = (1.0 - it.epoch_detail) / speed_e

        rate = it.epoch_detail
        marks = '#' * int(rate * bar_length)
        out.write('{}[{}{}] {:6.2%}\n'.format(
            title, marks, '.' * (bar_length - len(marks)), rate))

        if hasattr(it, '_epoch_size'):
            out.write('{:10} / {} iterations\n'
                      .format(it.current_position, it._epoch_size))
        else:
            out.write('{:10} iterations\n'.format(it.current_position))
        out.write('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                  .format(speed_t,
                          datetime.timedelta(seconds=estimated_time)))

        # move the cursor to the head of the progress bar
        if os.name == 'nt':
            set_console_cursor_position(0, -3)
        else:
            out.write('\033[3A')
        if hasattr(out, 'flush'):
            out.flush()

        if len(recent_timing) > 100:
            del recent_timing[0]

    def close(self):
        out = self._out

        if os.name == 'nt':
            erase_console(0, 0)
        else:
            out.write('\033[J')
        if hasattr(out, 'flush'):
            out.flush()
