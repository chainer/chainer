import collections
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
        Mode=1: From (x,y) position down to the beginning of line.
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


class ProgressBar(object):

    def __init__(self, bar_length=None, out=None):
        self._bar_length = 50 if bar_length is None else bar_length
        self._out = sys.stdout if out is None else out
        self._recent_timing = collections.deque([], maxlen=100)

    def update_speed(self, iteration, epoch_detail):
        now = time.time()
        self._recent_timing.append((iteration, epoch_detail, now))
        old_t, old_e, old_sec = self._recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (iteration - old_t) / span
            speed_e = (epoch_detail - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')
        return speed_t, speed_e

    def get_lines(self):
        raise NotImplementedError

    def update(self):
        self.erase_console()

        lines = self.get_lines()
        for line in lines:
            self._out.write(line)

        self.move_cursor_up(len(lines))
        self.flush()

    def close(self):
        self.erase_console()
        self.flush()

    def erase_console(self):
        if os.name == 'nt':
            erase_console(0, 0)
        else:
            self._out.write('\033[J')

    def move_cursor_up(self, n):
        # move the cursor to the head of the progress bar
        if os.name == 'nt':
            set_console_cursor_position(0, - n)
        else:
            self._out.write('\033[{:d}A'.format(n))

    def flush(self):
        if hasattr(self._out, 'flush'):
            self._out.flush()
