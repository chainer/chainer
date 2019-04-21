import math


def check(app, what, name, obj, options, lines):
    ctx = DocstringCheckContext(app, what, name, obj, options, lines)

    if what in ('function', 'method'):
        _docstring_check_returns_indent(ctx)


class DocstringCheckContext(object):
    def __init__(self, app, what, name, obj, options, lines):
        self.app = app
        self.what = what
        self.name = name
        self.obj = obj
        self.options = options
        self.lines = lines

        self.iline = 0

    def nextline(self):
        if self.iline >= len(self.lines):
            raise StopIteration
        line = self.lines[self.iline]
        self.iline += 1
        return line

    def error(self, msg, include_line=True, include_source=True):
        lines = self.lines
        iline = self.iline - 1
        msg = ('{}\n\n'
               'on {}'.format(msg, self.name))

        if include_line and 0 <= iline < len(lines):
            line = lines[iline]
            msg += '\n' + 'at line {}: "{}"\n'.format(iline, line)

        if include_source:
            msg += '\n'
            msg += 'docstring:\n'
            digits = int(math.floor(math.log10(len(lines)))) + 1
            linum_fmt = '{{:0{}d}} '.format(digits)
            for i, line in enumerate(lines):
                msg += linum_fmt.format(i) + line + '\n'
        raise InvalidDocstringError(msg, self, iline)


class InvalidDocstringError(Exception):
    def __init__(self, msg, ctx, iline):
        super(InvalidDocstringError, self).__init__(self, msg)
        self.msg = msg
        self.ctx = ctx
        self.iline = iline

    def __str__(self):
        return self.msg


def _docstring_check_returns_indent(ctx):
    # Seek the :returns: header
    try:
        line = ctx.nextline()
        while line != ':returns:':
            line = ctx.nextline()
    except StopIteration:
        return  # No `Returns` section

    # Skip empty lines and seek the first line of the content
    try:
        line = ctx.nextline()
        while not line:
            line = ctx.nextline()
    except StopIteration:
        ctx.error('`Returns` section has no content')

    # Find the indentation of the first line
    # (note: line should have at least one non-space character)
    nindent = next(i for i, c in enumerate(line) if c != ' ')

    # Check the indentation of the following lines
    try:
        line = ctx.nextline()
        while line.startswith(' '):
            if (not line.startswith(' ' * nindent) or
                    line[nindent:].startswith(' ')):
                ctx.error('Invalid indentation of `Returns` section')
            line = ctx.nextline()
    except StopIteration:
        pass
