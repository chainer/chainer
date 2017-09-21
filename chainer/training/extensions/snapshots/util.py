import os
import shutil
import tempfile


def save(filename, outdir, handler):
    prefix = 'tmp' + filename
    fd, tmppath = tempfile.mkstemp(prefix=prefix, dir=outdir)
    try:
        handler.save(tmppath)
    except Exception:
        os.close(fd)
        os.remove(tmppath)
        raise
    os.close(fd)
    shutil.move(tmppath, os.path.join(outdir, filename))
