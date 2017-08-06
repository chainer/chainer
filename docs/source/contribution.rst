.. _contrib:

Contribution Guide
==================

This is a guide for all contributions to Chainer.
The development of Chainer is running on `the official repository at GitHub <https://github.com/chainer/chainer>`_.
Anyone that wants to register an issue or to send a pull request should read through this document.

.. note::

   Many points of this document are updated at v2.
   We strongly recommend all contributors of v1 to read through the document again.

Classification of Contributions
-------------------------------

There are several ways to contribute to Chainer community:

1. Registering an issue
2. Sending a pull request (PR)
3. Sending a question/reply to `StackOverflow <https://stackoverflow.com/>`_ (with ``chainer`` tag) or `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_
4. Open-sourcing an external example
5. Writing a post about Chainer

This document mainly focuses on 1 and 2, though other contributions are also appreciated.


Development Cycle
-----------------

This section explains the development process of Chainer.
Before contributing to Chainer, it is strongly recommended to understand the development cycle.

Versioning
~~~~~~~~~~

The versioning of Chainer follows `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ and a part of `Semantic versioning <http://semver.org/>`_.
The version number consists of three or four parts: ``X.Y.Zw`` where ``X`` denotes the **major version**, ``Y`` denotes the **minor version**, ``Z`` denotes the **revision number**, and the optional ``w`` denotes the prelease suffix.
While the major, minor, and revision numbers follow the rule of semantic versioning, the pre-release suffix follows PEP 440 so that the version string is much friendly with Python eco-system.

**Note that a major update basically does not contain compatibility-breaking changes from the last release candidate (RC).**
This is not a strict rule, though; if there is a critical API bug that we have to fix for the major version, we may add breaking changes to the major version up.

As for the backward compatibility, see :ref:`compatibility`.


.. _contrib-release-cycle:

Release Cycle
~~~~~~~~~~~~~

Starting from v2.0.0, we are developing two tracks of versions at the same time.
The first one is the track of **stable versions**, which is a series of revision updates for the latest major version.
The second one is the track of **development versions**, which is a series of pre-releases for the upcoming major version.

Consider that ``X.0.0`` is the latest major version and ``Y.0.0``, ``Z.0.0`` are the succeeding major versions.
Then, the timeline of the updates is depicted by the following table.

========== =========== =========== ============
   Date       ver X       ver Y       ver Z
========== =========== =========== ============
  0 weeks    X.0.0rc1    --         --
  4 weeks    X.0.0       Y.0.0a1    --
  8 weeks    X.0.1       Y.0.0b1    --
 12 weeks    X.0.2       Y.0.0rc1   --
 16 weeks    --          Y.0.0      Z.0.0a1
========== =========== =========== ============

The dates shown in the left-most column are relative to the release of ``X.0.0rc1``.
In particular, each revision release is made four weeks after the previous one of the same major version, and the pre-release of the upcoming major version is made at the same time.

Note that the development of ``X.0.x`` stops at ``X.0.2``.
During the parallel development of ``Y.0.0`` and ``Z.0.0a1``, the version ``Y`` is treated as an **almost-stable version** and ``Z`` is treated as a development version.

If there is a critical bug found in ``X.0.2`` after stopping the development of version ``X``, we may release a hot-fix for this version at any time.

.. note::

   The release cycle of ``2.0.x`` and ``3.0.0x`` are slightly different from this table because we do not have ``3.0.0a1`` at the timing of the release of ``2.0.0``.
   In this case, the releases of ``3.0.0x`` are shifted four weeks behind the usual timeline, that is, ``3.0.0a1`` will be released at the same time with ``2.0.1``.

As you can see in the above table, we basically do not have any minor releases from v2.
All changes that add and/or modify APIs should be made by the pre-release updates.

We create a milestone for each upcoming release at GitHub.
The GitHub milestone is basically used for collecting the issues and PRs resolved in the release.

.. _contrib-git-branches:

Git Branches
~~~~~~~~~~~~

The ``master`` branch is used to develop pre-release versions.
It means that **alpha, beta, and RC updates are developed at the** ``master`` **branch**.
This branch contains the most up-to-date source tree that includes features newly added after the latest major version.

The stable version is developed at the individual branch named as ``vN`` where "N" reflects the version number (we call it a *versioned branch*).
For example, v3.0.0, v3.0.1, and v3.0.2 will be developed at the ``v3`` branch.

**Notes for contributors:**
When you send a pull request, you basically have to send it to the ``master`` branch.
If the change can also be applied to the stable version, a core team member will apply the same change to the stable version so that the change is also included in the next revision update.

If the change is only applicable to the stable version and not to the ``master`` branch, please send it to the versioned branch.
We basically only accept changes to the latest versioned branch (where the stable version is developed) unless the fix is critical.

*Note: a change that can be applied to both branches should be sent to the* ``master`` *branch.*
*Each release of the stable version is also merged to the development version so that the change is also reflected to the next major version.*

Issues and Pull Requests
------------------------

In this section, we explain how to file issues and send pull requests (PRs).

Issue/PR Labels
~~~~~~~~~~~~~~~

Issues and PRs are labeled by the following tags:

* **Bug**: bug reports (issues) and bug fixes (PRs)
* **Enhancement**: implementation improvements without breaking the interface
* **Feature**: feature requests (issues) and their implementations (PRs)
* **NoCompat**: disrupts backward compatibility
* **Test**: test fixes and updates
* **Document**: document fixes and improvements
* **Example**: fixes and improvements on the examples
* **Install**: fixes installation script
* **Contribution-Welcome**: issues that we request for contribution (only issues are categorized to this)
* **Other**: other issues and PRs

Multiple tags might be labeled to one issue/PR.
**Note that revision releases cannot include PRs in Feature and NoCompat categories.**

How to File an Issue
~~~~~~~~~~~~~~~~~~~~

On registering an issue, write precise explanations on how you want Chainer to be.
Bug reports must include necessary and sufficient conditions to reproduce the bugs.
Feature requests must include **what** you want to do (and **why** you want to do, if needed) with Chainer.
You can contain your thoughts on **how** to realize it into the feature requests, though **what** part is most important for discussions.

.. warning::

   If you have a question on usages of Chainer, it is highly recommended to send a post to `StackOverflow <https://stackoverflow.com/>`_ or `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_ instead of the issue tracker.
   The issue tracker is not a place to share knowledge on practices.
   We may suggest these places and immediately close how-to question issues.

How to Send a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you can write code to fix an issue, we encourage to send a PR.

First of all, before starting to write any code, do not forget to confirm the following points.

- Read through the :ref:`coding-guide` and :ref:`testing-guide`.
- Check the appropriate branch that you should send the PR following :ref:`contrib-git-branches`.
  If you do not have any idea about selecting a branch, please choose the ``master`` branch.

In particular, **check the branch before writing any code.**
The current source tree of the chosen branch is the starting point of your change.

After writing your code **(including unit tests and hopefully documentations!)**, send a PR on GitHub.
You have to write a precise explanation of **what** and **how** you fix;
it is the first documentation of your code that developers read, which is a very important part of your PR.

Once you send a PR, it is automatically tested on `Travis CI <https://travis-ci.org/chainer/chainer/>`_ for Linux and Mac OS X, and on `AppVeyor <https://ci.appveyor.com/project/chainer/chainer>`_ for Windows.
Your PR needs to pass at least the test for Linux on Travis CI.
After the automatic test passes, some of the core developers will start reviewing your code.
Note that this automatic PR test only includes CPU tests.

.. note::

   We are also running continuous integration with GPU tests for the ``master`` branch and the versioned branch of the latest major version.
   Since this service is currently running on our internal server, we do not use it for automatic PR tests to keep the server secure.

If you are planning to add a new feature or modify existing APIs, **it is recommended to open an issue and discuss the design first.**
The design discussion needs lower cost for the core developers than code review.
Following the consequences of the discussions, you can send a PR that is smoothly reviewed in a shorter time.

Even if your code is not complete, you can send a pull request as a *work-in-progress PR* by putting the ``[WIP]`` prefix to the PR title.
If you write a precise explanation about the PR, core developers and other contributors can join the discussion about how to proceed the PR.
WIP PR is also useful to have discussions based on a concrete code.


.. _coding-guide:

Coding Guidelines
-----------------

We use `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and a part of `OpenStack Style Guidelines <http://docs.openstack.org/developer/hacking/>`_ related to general coding style as our basic style guidelines.

To check your code, use ``autopep8`` and ``flake8`` command installed by ``hacking`` package::

  $ pip install autopep8 hacking
  $ autopep8 --global-config .pep8 path/to/your/code.py
  $ flake8 path/to/your/code.py

The ``autopep8`` supports automatically correct Python code to conform to the PEP 8 style guide::

  $ autopep8 --in-place --global-config .pep8 path/to/your/code.py

The ``flake8`` command lets you know the part of your code not obeying our style guidelines.
Before sending a pull request, be sure to check that your code passes the ``flake8`` checking.

Note that ``flake8`` command is not perfect.
It does not check some of the style guidelines.
Here is a (not-complete) list of the rules that ``flake8`` cannot check.

* Relative imports are prohibited. [H304]
* Importing non-module symbols is prohibited.
* Import statements must be organized into three parts: standard libraries, third-party libraries, and internal imports. [H306]

In addition, we restrict the usage of *shortcut symbols* in our code base.
They are symbols imported by packages and sub-packages of ``chainer``.
For example, ``chainer.Variable`` is a shortcut of ``chainer.variable.Variable``.
**It is not allowed to use such shortcuts in the Chainer library implementation.**
Note that you can still use them in ``tests`` and ``examples`` directories.
Also note that you should use shortcut names of CuPy APIs in Chainer implementation.

Once you send a pull request, your coding style is automatically checked by `Travis-CI <https://travis-ci.org/chainer/chainer/>`_.
The reviewing process starts after the check passes.


.. _testing-guide:

Unit Testing
------------

Testing is one of the most important part of your code.
You must write test cases and verify your implementation by following our testing guide.

Note that we are using nose and mock package for testing, so install them before writing your code::

  $ pip install nose mock

How to Run Tests
~~~~~~~~~~~~~~~~

You can run unit tests simply by running ``nosetests`` command at the repository root::

  $ nosetests

or specify the test script that you want to run::

  $ nosetests path/to/your/test.py

You can also run all unit tests under a specified directory::

  $ nosetests tests/chainer_tests/<directory name>

It requires CUDA by default.
In order to run unit tests that do not require CUDA, pass ``--attr='!gpu'`` option to the ``nosetests`` command::

  $ nosetests path/to/your/test.py --attr='!gpu'

Some GPU tests involve multiple GPUs.
If you want to run GPU tests with insufficient number of GPUs, specify the number of available GPUs by ``--eval-attr='gpu<N'`` where ``N`` is a concrete integer.
For example, if you have only one GPU, launch ``nosetests`` by the following command to skip multi-GPU tests::

  $ nosetests path/to/gpu/test.py --eval-attr='gpu<2'

Some tests spend too much time.
If you want to skip such tests, pass ``--attr='!slow'`` option to the ``nosetests`` command::

  $ nosetests path/to/your/test.py --attr='!slow'

If you modify the code related to existing unit tests, you must run appropriate commands and confirm that the tests pass.

Test File and Directory Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests are put into the ``tests/chainer_tests`` directory.
In order to enable test runner to find test scripts correctly, we are using special naming convention for the test subdirectories and the test scripts.

* The name of each subdirectory of ``tests`` must end with the ``_tests`` suffix.
* The name of each test script must start with the ``test_`` prefix.

When we write a test for a module, we use the appropriate path and file name for the test script whose correspondence to the tested module is clear.
For example, if you want to write a test for a module ``chainer.x.y.z``, the test script must be located at ``tests/chainer_tests/x_tests/y_tests/test_z.py``.

How to Write Tests
~~~~~~~~~~~~~~~~~~

There are many examples of unit tests under the ``tests`` directory, so reading some of them is a good and recommended way to learn how to write tests for Chainer.
They simply use the ``unittest`` package of the standard library, while some tests are using utilities from :mod:`chainer.testing`.

Even if your patch includes GPU-related code, your tests should not fail without GPU capability.
Test functions that require CUDA must be tagged by ``chainer.testing.attr.gpu`` decorator::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.gpu
      def test_my_gpu_func(self):
          ...

The functions tagged by the ``gpu`` decorator are skipped if ``--attr='!gpu'`` is given.
We also have the ``chainer.testing.attr.cudnn`` decorator to let ``nosetests`` know that the test depends on cuDNN.

The test functions decorated by ``gpu`` must not depend on multiple GPUs.
In order to write tests for multiple GPUs, use ``chainer.testing.attr.multi_gpu()`` decorator instead::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.multi_gpu(2)  # specify the number of required GPUs here
      def test_my_two_gpu_func(self):
          ...

If your test requires too much time, add ``chainer.testing.attr.slow`` decorator.
The test functions decorated by ``slow`` are skipped if ``--attr='!slow'`` is given::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.slow
      def test_my_slow_func(self):
          ...

.. note::
   If you want to specify more than two attributes, separate them with a comma such as ``--attr='!gpu,!slow'``.
   See detail in `the document of nose <https://nose.readthedocs.io/en/latest/plugins/attrib.html#simple-syntax>`_.

Once you send a pull request, your code is automatically tested by `Travis-CI <https://travis-ci.org/chainer/chainer/>`_ **with --attr='!gpu,!slow' option**.
Since Travis-CI does not support CUDA, we cannot check your CUDA-related code automatically.
The reviewing process starts after the test passes.
Note that reviewers will test your code without the option to check CUDA-related code.

.. note::
   Some of numerically unstable tests might cause errors irrelevant to your changes.
   In such a case, we ignore the failures and go on to the review process, so do not worry about it!
