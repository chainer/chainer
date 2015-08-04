import datetime
import functools


def time(enable=True):
    def _time(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            assert len(args) > 0
            instance = args[0]
            class_name = instance.__class__.__name__
            if enable:
                print('Start\t{}.{}'.format(class_name, f.__name__))
                start = datetime.datetime.now()

            ret = f(*args, **kwargs)
            if enable:
                finish = datetime.datetime.now()
                elapsed = (finish - start).microseconds
                print('Finish\t{}.{}\t{} us'.format(
                    class_name, f.__name__, elapsed))
            return ret
        return wrapper
    return _time
