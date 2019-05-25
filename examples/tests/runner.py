import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))

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


class ExampleRunner(object):

    """Example runner.

    A single runner can run multiple script commands.
    Each script will be altered with the specified templates.
    A runner creates a temporary directory.
    All the runs are executed within the temporary directory as the current
    directory.
    """

    def __init__(self, *, template_dir):
        if template_dir is not None:
            template_dir = os.path.join(REPO_ROOT, template_dir)
            assert os.path.isdir(template_dir), template_dir

        self.contexts = None
        self.template_dir = template_dir

    def __enter__(self):
        contexts = []

        tempdir_ = tempdir()
        tempd = tempdir_.__enter__()
        contexts.append(tempdir_)

        chdir_ = chdir(tempd)
        chdir_.__enter__()
        contexts.append(chdir_)

        self.contexts = contexts
        return self

    def __exit__(self, typ, value, traceback):
        for c in reversed(self.contexts):
            c.__exit__(typ, value, traceback)
        self.contexts = None

    def run(self, script_path, args):
        # Runs a command.
        assert isinstance(script_path, str), type(script_path)
        assert isinstance(args, list), type(args)
        template_dir = self.template_dir
        script_path = os.path.join(REPO_ROOT, script_path)

        temp_script_path = os.path.join(
            os.path.dirname(script_path),
            '.test_' + os.path.basename(script_path))

        try:
            # Modify the script file by applying templates.
            with open(script_path, 'r') as file_in:
                with open(temp_script_path, 'w') as file_out:
                    _filter_script_file(
                        file_in,
                        file_out,
                        template_dir)

            command = [
                sys.executable,
                temp_script_path] + args

            # Run the command.
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdoutdata, stderrdata = proc.communicate()

        finally:
            os.remove(temp_script_path)

        if proc.returncode != 0:
            err_fmt = '''\
Script exited with {returncode}.
== command ==
{command}
== stdout ==
{stdout}
== stderr ==
{stderr}
'''
            err = err_fmt.format(
                returncode=proc.returncode,
                command=' '.join(command),
                stdout=stdoutdata.decode('utf8'),
                stderr=stderrdata.decode('utf8'))
            raise RuntimeError(err)


def _filter_script_file(file_in, file_out, template_dir):
    # Applies the templates to a script file.
    #
    # If template_dir is None, it simply copies line-by-line.

    regex = re.compile(r'^ *# \.\.(?P<name>[_a-zA-Z0-1]+)$')
    for line in file_in:
        if template_dir is None:
            m = None
        else:
            m = regex.match(line)

        if m:
            template_name = m.group('name')
            template_path = os.path.join(template_dir, template_name)
            with open(template_path, 'r') as file_template:
                for template_line in file_template:
                    file_out.write(template_line)
        else:
            file_out.write(line)
