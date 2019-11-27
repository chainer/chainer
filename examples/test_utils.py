import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile


class OutputEvaluator(object):
    def check(self, outdata):
        raise NotImplementedError()


class TemplateOutputEvaluator(OutputEvaluator):

    def __init__(self, template, **checks):
        self.template = template
        self.checks = checks

    def check(self, outdata):
        template = self.template
        checks = self.checks

        lines = outdata.split(b'\n')
        tmpl_lines = template.split(b'\n')

        # Collect placeholders included in the template
        temps = {}
        for iline, tmpl_line in enumerate(tmpl_lines):
            for m in re.finditer(rb'{(?P<key>[_a-zA-Z0-9]+) *}', tmpl_line):
                key = m.groupdict()['key'].decode('utf8')
                assert key not in temps
                temps[key] = (m, iline)

        # Keys of the placeholders and the checks must match.
        assert set(temps.keys()) == set(checks.keys())

        # Evaluate the checks
        for key, (m, iline) in temps.items():
            line = lines[iline]
            c = checks[key]
            typ, checkfunc = c
            i1, i2 = m.span()
            i2 = min(i2, len(line))
            s = line[i1:i2]
            if typ is float:
                value = float(s)
                if not checkfunc(value):
                    raise RuntimeError('Check fail: key={}'.format(key))
            else:
                raise TypeError('Invalid check type: {}'.format(typ))


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

EXAMPLES_ROOT = os.path.join(REPO_ROOT, 'examples')


@contextlib.contextmanager
def tempdir(**kwargs):
    # A context manager that defines a lifetime of a temporary directory.

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


@contextlib.contextmanager
def chdir(path):
    # A context manager that changes the current directory temporarily.

    old_chdir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_chdir)


class ReplacementFileCorrectnessError(Exception):
    def __init__(
            self,
            message,
            *,
            orig_path,
            replace_path,
            orig_line_num=None,
            replace_line_num=None,
            orig_line=None,
            replace_line=None):

        infos = [
            'Original file: {}'.format(orig_path),
            'Replacement file: {}'.format(replace_path),
        ]
        if orig_line_num is not None:
            infos.append(
                'Line number in the original file: {}'.format(orig_line_num))
        if replace_line_num is not None:
            infos.append(
                'Line number in the replacement file: {}'.format(
                    replace_line_num))
        if orig_line is not None:
            infos.append(
                'Original line: [{}]'.format(orig_line))
        if replace_line is not None:
            infos.append(
                'Replacement line: [{}]\n'.format(replace_line))
        message_ = (
            'Replacement file correctness check failed: {message}\n\n'
            'If you\'re seeing this error message, it\'s likely that you '
            'edited a file within the example directory but did not edit the '
            'matching replacement file which is used for testing. '
            'Please ensure the two files are synchronized.\n\n'
            '{infos}'.format(
                message=message,
                infos='\n'.join(infos)))
        super().__init__(message_)


class ExampleRunner(object):

    """Example runner.

    A single runner can run multiple script commands.
    A runner creates a temporary directory and files in the respective example
    directory are copied there.
    All the runs are executed within the temporary directory as the current
    directory.
    """

    contexts = None
    work_dir = None  # assigned on __enter__

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __enter__(self):
        contexts = []

        # Create a temporary directory.
        tempd = tempdir()
        work_dir = tempd.__enter__()
        contexts.append(tempd)

        # Change the current directory.
        chdir_ = chdir(work_dir)
        chdir_.__enter__()
        contexts.append(chdir_)

        self.work_dir = work_dir
        self.contexts = contexts

        # Initialize the work directory.
        self._init_work_dir()
        return self

    def __exit__(self, typ, value, traceback):
        for c in reversed(self.contexts):
            c.__exit__(typ, value, traceback)
        self.contexts = None

    def _init_work_dir(self):
        # Copies files in the directory.
        # If a replacement file exists for each file, copy it instead of the
        # original file.
        # Correctness of the replacement files are also checked.

        root_dir = self.root_dir
        work_dir = self.work_dir
        assert os.path.isdir(root_dir), root_dir
        assert os.path.isdir(work_dir), work_dir

        replace_dir = os.path.join(root_dir, '.testdata', 'replacements')
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip directories starting with '.'.
            dirnames[:] = [dn for dn in dirnames if not dn.startswith('.')]

            if dirpath == root_dir:
                dir_relpath = ''
            else:
                dir_relpath = os.path.relpath(dirpath, root_dir)

            for filename in filenames:
                relpath = os.path.join(dir_relpath, filename)
                orig_path = os.path.join(root_dir, relpath)
                dst_path = os.path.join(work_dir, relpath)
                # Check to see if the replace file exists.
                replace_path = os.path.join(replace_dir, relpath)
                if os.path.isfile(replace_path):
                    # The replace file exists: check correctness of the file
                    # comparing with the original file.
                    self._check_replace_file_correct(orig_path, replace_path)
                    # Copy the replace file.
                    self._copyfile(replace_path, dst_path)
                else:
                    # The replace file does not exist: copy the original file.
                    self._copyfile(orig_path, dst_path)

    def _copyfile(self, src_path, dst_path):
        dirpath = os.path.dirname(dst_path)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        shutil.copyfile(src_path, dst_path)

    def _check_replace_file_correct(self, orig_path, replace_path):
        """Checks correctness of the replacement file comparing with the
        original file."""

        MARKER_BEGIN = '# BEGIN ADDITIONAL TEST CODE'
        MARKER_END = '# END ADDITIONAL TEST CODE'

        # Read lines from both files.
        with open(orig_path, 'r') as orig_file:
            orig_lines = orig_file.readlines()
        with open(replace_path, 'r') as replace_file:
            replace_lines = replace_file.readlines()

        j = 0  # line number (replace)
        for i, orig_line in enumerate(orig_lines):
            if len(replace_lines) <= j:
                raise ReplacementFileCorrectnessError(
                    'Replacement file has less lines than the original.',
                    orig_path=orig_path,
                    replace_path=replace_path)
            replace_line = replace_lines[j]
            j += 1

            # Check if the line is a starting marker comment.
            # Marker line can start with arbitrary number of spaces (indent).
            if (replace_line.endswith(MARKER_BEGIN + '\n')
                    and all(' ' == c
                            for c in replace_line[:-len(MARKER_BEGIN)-1])):
                # Starting marker is found: find the corresponding ending
                # marker and retrieve the next line.
                indent_count = len(replace_line) - len(MARKER_BEGIN) - 1
                end_marker_line = ' ' * indent_count + MARKER_END + '\n'
                j_ = j
                while True:
                    if len(replace_lines) <= j_:
                        raise ReplacementFileCorrectnessError(
                            'Matching ending marker could not be found in a '
                            'replacement file.',
                            orig_path=orig_path,
                            replace_path=replace_path,
                            orig_line_num=i,
                            replace_line_num=j,
                            orig_line=orig_line.rstrip('\n'),
                            replace_line=replace_line.rstrip('\n'))
                    if replace_lines[j_] == end_marker_line:
                        break
                    j_ += 1
                j = j_ + 1

                replace_line = replace_lines[j]
                j += 1

            # Compare the next non-marked lines.
            if orig_line != replace_line:
                raise ReplacementFileCorrectnessError(
                    'Line mismatch between the original and the replacement '
                    'file.',
                    orig_path=orig_path,
                    replace_path=replace_path,
                    orig_line_num=i,
                    replace_line_num=j,
                    orig_line=orig_line.rstrip('\n'),
                    replace_line=replace_line.rstrip('\n'))

        if j != len(replace_lines):
            raise ReplacementFileCorrectnessError(
                'Replacement file has more lines than the original.',
                orig_path=orig_path,
                replace_path=replace_path)

    def run(self, script_name, args, *, output_evaluator=None):
        # Runs a command.

        assert self.contexts is not None, (
            '__enter__ has not been called on the example runner.')
        assert isinstance(script_name, str), type(script_name)
        assert isinstance(args, list), type(args)
        assert (output_evaluator is None
                or isinstance(output_evaluator, OutputEvaluator)), (
                    type(output_evaluator))
        work_dir = self.work_dir
        script_path = os.path.join(work_dir, script_name)
        assert os.path.isfile(script_path), script_path

        command = [
            sys.executable,
            script_path] + args

        # Run the command.
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdoutdata, stderrdata = proc.communicate()

        def fail(msg):
            err_fmt = '''\
{message}
== command ==
{command}
== stdout ==
{stdout}
== stderr ==
{stderr}
'''
            err = err_fmt.format(
                message=msg,
                command=' '.join(command),
                stdout=stdoutdata.decode('utf8'),
                stderr=stderrdata.decode('utf8'))
            raise RuntimeError(err)

        if proc.returncode != 0:
            fail('Script exited with {}.'.format(proc.returncode))

        if output_evaluator is not None:
            try:
                output_evaluator.check(stdoutdata)
            except RuntimeError as e:
                fail(
                    'Script output does not meet expectation:\n'
                    '{}'.format(e))
