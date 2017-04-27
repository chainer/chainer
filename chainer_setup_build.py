from __future__ import print_function
from distutils import ccompiler
from distutils import errors
from distutils import msvccompiler
from distutils import sysconfig
from distutils import unixccompiler
import os
from os import path
import re
import subprocess
import sys

import pkg_resources
import setuptools
from setuptools.command import build_ext

from install import build
from install import utils


required_cython_version = pkg_resources.parse_version('0.24.0')

MODULES = [
    {
        'name': 'cuda',
        'file': [
            'cupy.core.core',
            'cupy.core.flags',
            'cupy.core.internal',
            'cupy.cuda.cublas',
            'cupy.cuda.curand',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.pinned_memory',
            'cupy.cuda.profiler',
            'cupy.cuda.nvtx',
            'cupy.cuda.function',
            'cupy.cuda.runtime',
            'cupy.util',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_profiler_api.h',
            'cuda_runtime.h',
            'curand.h',
            'nvToolsExt.h',
        ],
        'libraries': [
            'cublas',
            'cuda',
            'cudart',
            'curand',
            'nvToolsExt',
        ],
        'check_method': build.check_cuda_version,
    },
    {
        'name': 'cudnn',
        'file': [
            'cupy.cuda.cudnn',
        ],
        'include': [
            'cudnn.h',
        ],
        'libraries': [
            'cudnn',
        ],
        'check_method': build.check_cudnn_version,
    },
    {
        'name': 'cusolver',
        'file': [
            'cupy.cuda.cusolver',
        ],
        'include': [
            'cusolverDn.h',
        ],
        'libraries': [
            'cusolver',
        ],
        'check_method': build.check_cusolver_version,
    },
    {
        # The value of the key 'file' is a list that contains extension names
        # or tuples of an extension name and a list of other souces files
        # required to build the extension such as .cpp files and .cu files.
        #
        #   <extension name> | (<extension name>, a list of <other source>)
        #
        # The extension name is also interpreted as the name of the Cython
        # source file required to build the extension with appending '.pyx'
        # file extension.
        'name': 'thrust',
        'file': [
            ('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu']),
        ],
        'include': [
            'thrust/device_ptr.h',
            'thrust/sort.h',
        ],
        'libraries': [
            'cudart',
        ],
        'check_method': build.check_cuda_version,
    }
]

if sys.platform == 'win32':
    mod_cuda = MODULES[0]
    mod_cuda['libraries'].remove('nvToolsExt')
    if utils.search_on_path(['nvToolsExt64_1.dll']) is None:
        mod_cuda['file'].remove('cupy.cuda.nvtx')
        mod_cuda['include'].remove('nvToolsExt.h')
        utils.print_warning(
            'Cannot find nvToolsExt. nvtx was disabled.')
    else:
        mod_cuda['libraries'].append('nvToolsExt64_1')


def ensure_module_file(file):
    if isinstance(file, tuple):
        return file
    else:
        return (file, [])


def module_extension_name(file):
    return ensure_module_file(file)[0]


def module_extension_sources(file, use_cython, no_cuda):
    pyx, others = ensure_module_file(file)
    ext = '.pyx' if use_cython else '.cpp'
    pyx = path.join(*pyx.split('.')) + ext

    # If CUDA SDK is not available, remove CUDA C files from extension sources
    # and use stubs defined in header files.
    if no_cuda:
        others1 = []
        for source in others:
            base, ext = os.path.splitext(source)
            if ext == '.cu':
                continue
            others1.append(source)
        others = others1

    return [pyx] + others


def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'


def check_library(compiler, includes=(), libraries=(),
                  include_dirs=(), library_dirs=()):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main(int argc, char* argv[]) {return 0;}'
    try:
        # We need to try to build a shared library because distutils
        # uses different option to build an executable and a shared library.
        # Especially when a user build an executable, distutils does not use
        # LDFLAGS environment variable.
        build.build_shlib(compiler, source, libraries,
                          include_dirs, library_dirs)
    except Exception as e:
        print(e)
        return False
    return True


def make_extensions(options, compiler, use_cython):
    """Produce a list of Extension instances which passed to cythonize()."""

    no_cuda = options['no_cuda']
    settings = build.get_compiler_setting()

    include_dirs = settings['include_dirs']

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]
    if sys.platform != 'win32':
        settings['runtime_library_dirs'] = settings['library_dirs']
    if sys.platform == 'darwin':
        args = settings.setdefault('extra_link_args', [])
        args.append(
            '-Wl,' + ','.join('-rpath,' + p
                              for p in settings['library_dirs']))
        # -rpath is only supported when targetting Mac OS X 10.5 or later
        args.append('-mmacosx-version-min=10.5')

    # This is a workaround for Anaconda.
    # Anaconda installs libstdc++ from GCC 4.8 and it is not compatible
    # with GCC 5's new ABI.
    settings['define_macros'].append(('_GLIBCXX_USE_CXX11_ABI', '0'))

    if options['linetrace']:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
        settings['define_macros'].append(('CYTHON_TRACE_NOGIL', '1'))
    if no_cuda:
        settings['define_macros'].append(('CUPY_NO_CUDA', '1'))

    ret = []
    for module in MODULES:
        print('Include directories:', settings['include_dirs'])
        print('Library directories:', settings['library_dirs'])

        if not no_cuda:
            if not check_library(compiler,
                                 includes=module['include'],
                                 include_dirs=settings['include_dirs']):
                utils.print_warning(
                    'Include files not found: %s' % module['include'],
                    'Skip installing %s support' % module['name'],
                    'Check your CFLAGS environment variable')
                continue

            if not check_library(compiler,
                                 libraries=module['libraries'],
                                 library_dirs=settings['library_dirs']):
                utils.print_warning(
                    'Cannot link libraries: %s' % module['libraries'],
                    'Skip installing %s support' % module['name'],
                    'Check your LDFLAGS environment variable')
                continue

            if 'check_method' in module and \
               not module['check_method'](compiler, settings):
                continue

        s = settings.copy()
        if not no_cuda:
            s['libraries'] = module['libraries']

        if module['name'] == 'cusolver':
            args = s.setdefault('extra_link_args', [])
            # openmp is required for cusolver
            if compiler.compiler_type == 'unix' and sys.platform != 'darwin':
                # In mac environment, openmp is not required.
                args.append('-fopenmp')
            elif compiler.compiler_type == 'msvc':
                args.append('/openmp')

        for f in module['file']:
            name = module_extension_name(f)
            sources = module_extension_sources(f, use_cython, no_cuda)
            extension = setuptools.Extension(name, sources, **s)
            ret.append(extension)

    return ret


def parse_args():
    cupy_profile = '--cupy-profile' in sys.argv
    if cupy_profile:
        sys.argv.remove('--cupy-profile')
    cupy_coverage = '--cupy-coverage' in sys.argv
    if cupy_coverage:
        sys.argv.remove('--cupy-coverage')
    no_cuda = '--cupy-no-cuda' in sys.argv
    if no_cuda:
        sys.argv.remove('--cupy-no-cuda')

    arg_options = {
        'profile': cupy_profile,
        'linetrace': cupy_coverage,
        'annotate': cupy_coverage,
        'no_cuda': no_cuda,
    }
    if check_readthedocs_environment():
        arg_options['no_cuda'] = True
    return arg_options


def check_cython_version():
    try:
        import Cython
        cython_version = pkg_resources.parse_version(Cython.__version__)
        return cython_version >= required_cython_version
    except ImportError:
        return False


def cythonize(extensions, arg_options):
    import Cython.Build

    directive_keys = ('linetrace', 'profile')
    directives = {key: arg_options[key] for key in directive_keys}

    cythonize_option_keys = ('annotate',)
    cythonize_options = {key: arg_options[key]
                         for key in cythonize_option_keys}

    return Cython.Build.cythonize(
        extensions, verbose=True,
        compiler_directives=directives, **cythonize_options)


def check_extensions(extensions):
    for x in extensions:
        for f in x.sources:
            if not path.isfile(f):
                msg = ('Missing file: %s\n' % f +
                       'Please install Cython. ' +
                       'Please also check the version of Cython.\n' +
                       'See http://docs.chainer.org/en/stable/install.html')
                raise RuntimeError(msg)


def get_ext_modules():
    arg_options = parse_args()
    print('Options:', arg_options)

    # We need to call get_config_vars to initialize _config_vars in distutils
    # see #1849
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)

    use_cython = check_cython_version()
    extensions = make_extensions(arg_options, compiler, use_cython)

    if use_cython:
        extensions = cythonize(extensions, arg_options)

    check_extensions(extensions)
    return extensions


def _nvcc_gencode_options(cuda_version):
    """Returns NVCC GPU code generation options."""
    arch = ['sm_30', 'sm_32', 'sm_35', 'sm_37', 'sm_50', 'sm_52']
    if cuda_version >= 7000:
        arch += ['sm_53']
    if cuda_version >= 8000:
        arch += ['sm_60', 'sm_61', 'sm_62']

    return ['--gpu-architecture=compute_30',
            '--gpu-code=compute_30,{}'.format(','.join(arch))]


def _escape(str):
    return str.replace('\\', r'\\').replace('"', r'\"')


class _UnixCCompiler(unixccompiler.UnixCCompiler):
    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.append('.cu')

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For ordinal extension source files, just call the super class method.
        if os.path.splitext(src)[1] != '.cu':
            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        # For CUDA C source files, compile them with NVCC.
        _compiler_so = self.compiler_so
        try:
            nvcc_path = build.get_nvcc_path()
            self.set_executable('compiler_so', nvcc_path)

            cflags = ''
            if 'CFLAGS' in os.environ:
                cflags = cflags + ' ' + os.environ['CFLAGS']
            if 'CPPFLAGS' in os.environ:
                cflags = cflags + ' ' + os.environ['CPPFLAGS']

            compiler_options = '-fPIC'
            if cflags != '':
                compiler_options += ' ' + cflags
            compiler_options = _escape(compiler_options)

            cuda_version = build.get_cuda_version()
            postargs = _nvcc_gencode_options(cuda_version) + [
                '-O2', '--compiler-options="{}"'.format(compiler_options)]
            print('NVCC options:', postargs)

            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, postargs, pp_opts)
        finally:
            self.compiler_so = _compiler_so


def _split(lst, fn):
    then_list = []
    else_list = []
    for x in lst:
        if fn(x):
            then_list.append(x)
        else:
            else_list.append(x)
    return then_list, else_list


class _MSVCCompiler(msvccompiler.MSVCCompiler):
    _cu_extensions = ['.cu']

    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.extend(_cu_extensions)

    def _compile_cu(self, sources, output_dir=None, macros=None,
                    include_dirs=None, debug=0, extra_preargs=None,
                    extra_postargs=None, depends=None):
        # Compile CUDA C files, mainly derived from UnixCCompiler._compile().

        macros, objects, extra_postargs, pp_opts, _build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)

        compiler_so = [build.get_nvcc_path()]
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        postargs = _nvcc_gencode_options() + ['-O2']

        cflags = ''
        if 'CFLAGS' in os.environ:
            cflags = cflags + ' ' + os.environ['CFLAGS']
        if 'CPPFLAGS' in os.environ:
            cflags = cflags + ' ' + os.environ['CPPFLAGS']
        if cflags != '':
            postargs += [cflags]
        print('NVCC options:', postargs)

        for obj in objects:
            try:
                src, ext = _build[obj]
            except KeyError:
                continue
            try:
                self.spawn(compiler_so + cc_args + [src, '-o', obj] + postargs)
            except errors.DistutilsExecError as e:
                raise errors.CompileError(e.message)

        return objects

    def compile(self, sources, **kwargs):
        sources_cu, sources_base = _split(
            sources, lambda x: os.path.splitext(x)[1] == '.cu')

        # Compile source files other than CUDA C ones.
        super = msvccompiler.MSVCCompiler
        objects_base = super.compile(self, sources_base, **kwargs)

        # Compile CUDA C files
        objects_cu = self._compile_cu(sources_cu, *kwargs)

        # Return compiled object filenames.
        return objects_base + objects_cu


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if build.get_nvcc_path() is not None:
            def wrap_new_compiler(func):
                def _wrap_new_compiler(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except errors.DistutilsPlatformError:
                        if not sys.platform == 'win32':
                            CCompiler = _UnixCCompiler
                        else:
                            CCompiler = _MSVCCompiler
                        return CCompiler(
                            None, kwargs['dry_run'], kwargs['force'])
                return _wrap_new_compiler
            ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
            # Intentionally causes DistutilsPlatformError in
            # ccompiler.new_compiler() function to hook.
            self.compiler = 'nvidia'
        build_ext.build_ext.run(self)
