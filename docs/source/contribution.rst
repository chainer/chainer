.. _contrib:

Contribution Guide
==================

This is a guide for all contributions to Chainer.
The development of Chainer is running on `the official repository at GitHub <https://github.com/chainer/chainer>`_.
Anyone that wants to register an issue or to send a pull request should read through this document.

.. note::

   Many points of this documentation are updated at v2.
   We strongly recommend all contributors of v1 to read through the documentation again.

Classification of Contributions
-------------------------------

There are several ways to contribute to Chainer community:

1. Registering an issue
2. Sending a pull request (PR)
3. Sending a question/reply to `StackOverflow <https://stackoverflow.com/>`_ (with ``chainer`` tag) or `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_
4. Open-sourcing an external example
5. Writing a post about Chainer

This documentation mainly focuses on 1 and 2, though other contributions are also appreciated.


Development Cycle
-----------------

This section explains the development process of Chainer.
Before contributing to Chainer, it is strongly recommended to understand the development cycle.

Versioning
~~~~~~~~~~

The versioning of Chainer follows `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ and a part of `Semantic versioning <https://semver.org/>`_.
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
  8 weeks    X.1.0*      Y.0.0b1    --
 12 weeks    X.2.0*      Y.0.0rc1   --
 16 weeks    --          Y.0.0      Z.0.0a1
========== =========== =========== ============

(* These might be revision releases)

The dates shown in the left-most column are relative to the release of ``X.0.0rc1``.
In particular, each revision/minor release is made four weeks after the previous one of the same major version, and the pre-release of the upcoming major version is made at the same time.
Whether these releases are revision or minor is determined based on the contents of each update.

Note that there are only three stable releases for the versions ``X.x.x``.
During the parallel development of ``Y.0.0`` and ``Z.0.0a1``, the version ``Y`` is treated as an **almost-stable version** and ``Z`` is treated as a development version.

If there is a critical bug found in ``X.x.x`` after stopping the development of version ``X``, we may release a hot-fix for this version at any time.

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

If you want to make a new feature of the ``master`` branch available in the current stable version, please send a *backport PR* to the stable version (the latest ``vN`` branch).
See the next section for details.

*Note: a change that can be applied to both branches should be sent to the* ``master`` *branch.*
*Each release of the stable version is also merged to the development version so that the change is also reflected to the next major version.*

Feature Backport PRs
~~~~~~~~~~~~~~~~~~~~

We basically do not backport any new features of the development version to the stable versions.
If you desire to include the feature to the current stable version and you can work on the backport work, we welcome such a contribution.
In such a case, you have to send a backport PR to the latest ``vN`` branch.
**Note that we do not accept any feature backport PRs to older versions because we are not running quality assurance workflows (e.g. CI) for older versions so that we cannot ensure that the PR is correctly ported.**

There are some rules on sending a backport PR.

- Start the PR title from the prefix **[backport]**.
- Clarify the original PR number in the PR description (something like "This is a backport of #XXXX").
- (optional) Write to the PR description the motivation of backporting the feature to the stable version.

Please follow these rules when you create a feature backport PR.

Note: PRs that do not include any changes/additions to APIs (e.g. bug fixes, documentation improvements) are usually backported by core dev members.
It is also appreciated to make such a backport PR by any contributors, though, so that the overall development proceeds more smoothly!

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
* **Document**: documentation fixes and improvements
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

.. note::

   Coding guidelines are updated at v3.0.
   Those who have contributed to older versions should read the guidelines again.

We use `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and a part of `OpenStack Style Guidelines <https://docs.openstack.org/developer/hacking/>`_ related to general coding style as our basic style guidelines.

You can use ``autopep8`` and ``flake8`` commands to check your code.

In order to avoid confusion from using different tool versions, we pin the versions of those tools.
Install them with the following command (from within the top directory of Chainer repository)::

  $ pip install -e '.[stylecheck]'

And check your code with::

  $ autopep8 path/to/your/code.py
  $ flake8 path/to/your/code.py

The ``autopep8`` supports automatically correct Python code to conform to the PEP 8 style guide::

  $ autopep8 --in-place path/to/your/code.py

The ``flake8`` command lets you know the part of your code not obeying our style guidelines.
Before sending a pull request, be sure to check that your code passes the ``flake8`` checking.

Note that ``flake8`` command is not perfect.
It does not check some of the style guidelines.
Here is a (not-complete) list of the rules that ``flake8`` cannot check.

* Relative imports are prohibited. [H304]
* Importing non-module symbols is prohibited.
* Import statements must be organized into three parts: standard libraries, third-party libraries, and internal imports. [H306]

In addition, we restrict the usage of *shortcut aliases* in any global-scope code.
In particular, you cannot use shortcut aliases to designate a parent class in global-scope class definitions.
When you want to make a class inheriting another class defined in another module, you have to spell out the full module name instead of importing a module that provides an alias.

For example, the following code is not allowed.

.. code-block:: py

   import chainer

   class MyLink(chainer.Link): ...

Instead, import ``chainer.link`` and use that.

.. code-block:: py

   import chainer.link

   class MyLink(chainer.link.Link): ...

If you feel the code too verbose, you can also use ``from import`` or ``import as``.

.. code-block:: py

   from chainer import link

   class MyLink(link.Link): ...

.. note::

   From v3.0, we allow shortcut aliases used inside of functions and methods that are not called from any global scope code.
   For example, you can write ``chainer.Variable`` instead of ``chainer.variable.Variable`` inside of functions and methods.
   Use of such aliases is prohibited in the past for avoiding confusing errors related to cyclic dependencies;
   we relaxed the rule so that the library code looks similar to user code.

   When you use such shortcut aliases, please be careful with cyclic imports.
   One of the typical pitfalls is a way to import ``chainer.functions``.
   An import like ``import chainer.functions as F`` within modules under ``chainer.functions`` does not work.
   An import like ``from chainer import functions`` works well with Python 3, but does not with Python 2.
   We recommend you to use ``import chainer.functions`` and spell out like ``chainer.functions.foo`` in your methods.

Once you send a pull request, your coding style is automatically checked by `Travis-CI <https://travis-ci.org/chainer/chainer/>`_.
The reviewing process starts after the check passes.


.. _testing-guide:

Unit Testing
------------

Testing is one of the most important part of your code.
You must write test cases and verify your implementation by following our testing guide.

Note that we are using pytest and mock package for testing, so install them before writing your code::

  $ pip install pytest mock

How to Run Tests
~~~~~~~~~~~~~~~~

You can run unit tests simply by running ``python -m pytest`` command at the repository root::

  $ python -m pytest

or specify the test script that you want to run::

  $ python -m pytest path/to/your/test.py

You can also run all unit tests under a specified directory::

  $ python -m pytest tests/chainer_tests/<directory name>

It requires CUDA and cuDNN by default.
In order to run unit tests that do not require CUDA and cuDNN, use ``CHAINER_TEST_GPU_LIMIT=0`` environment variable and ``-m='not cudnn'`` option::

  $ export CHAINER_TEST_GPU_LIMIT=0
  $ python -m pytest path/to/your/test.py -m='not cudnn'

Some GPU tests involve multiple GPUs.
If you want to run GPU tests with insufficient number of GPUs, specify the number of available GPUs to ``CHAINER_TEST_GPU_LIMIT``.
For example, if you have only one GPU, launch ``pytest`` by the following command to skip multi-GPU tests::

  $ export CHAINER_TEST_GPU_LIMIT=1
  $ python -m pytest path/to/gpu/test.py

Some tests spend too much time.
If you want to skip such tests, pass ``-m='not slow'`` option to the command::

  $ python -m pytest path/to/your/test.py -m='not slow'

If you modify the code related to existing unit tests, you must run appropriate commands and confirm that the tests pass.

Test File and Directory Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests are put into the :tree:`tests/chainer_tests` directory.
In order to enable test runner to find test scripts correctly, we are using special naming convention for the test subdirectories and the test scripts.

* The name of each subdirectory of ``tests`` must end with the ``_tests`` suffix.
* The name of each test script must start with the ``test_`` prefix.

When we write a test for a module, we use the appropriate path and file name for the test script whose correspondence to the tested module is clear.
For example, if you want to write a test for a module ``chainer.x.y.z``, the test script must be located at ``tests/chainer_tests/x_tests/y_tests/test_z.py``.

How to Write Tests
~~~~~~~~~~~~~~~~~~

There are many examples of unit tests under the :tree:`tests` directory, so reading some of them is a good and recommended way to learn how to write tests for Chainer.
They simply use the :mod:`unittest` package of the standard library, while some tests are using utilities from :mod:`chainer.testing`.

In addition to the :ref:`coding-guide` mentioned above, the following rules are applied to the test code:

* All test classes must inherit from :class:`unittest.TestCase`.
* Use :mod:`unittest` features to write tests, except for the following cases:

    * Use ``assert`` statement instead of ``self.assert*`` methods (e.g., write ``assert x == 1`` instead of ``self.assertEqual(x, 1)``).
    * Use ``with pytest.raises(...):`` instead of ``with self.assertRaises(...):``.

.. note::

   We are incrementally applying the above style.
   Some existing tests may be using the old style (``self.assertRaises``, etc.), but all newly written tests should follow the above style.

Even if your patch includes GPU-related code, your tests should not fail without GPU capability.
Test functions that require CUDA must be tagged by ``chainer.testing.attr.gpu`` decorator::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.gpu
      def test_my_gpu_func(self):
          ...

The functions tagged by the ``gpu`` decorator are skipped if ``CHAINER_TEST_GPU_LIMIT=0`` environment variable is set.
We also have the ``chainer.testing.attr.cudnn`` decorator to let ``pytest`` know that the test depends on cuDNN.
The test functions decorated by ``cudnn`` are skipped if ``-m='not cudnn'`` is given.

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
The test functions decorated by ``slow`` are skipped if ``-m='not slow'`` is given::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.slow
      def test_my_slow_func(self):
          ...

.. note::
   If you want to specify more than two attributes, use ``and`` operator like ``-m='not cudnn and not slow'``.
   See detail in `the documentation of pytest <https://docs.pytest.org/en/latest/example/markers.html>`_.

Once you send a pull request, your code is automatically tested by `Travis-CI <https://travis-ci.org/chainer/chainer/>`_ **except for tests annotated with ``gpu``, ``multi_gpu`` and ``slow``**.
Since Travis-CI does not support CUDA, we cannot check your CUDA-related code automatically.
The reviewing process starts after the test passes.
Note that reviewers will test your code without the option to check CUDA-related code.

.. note::
   Some of numerically unstable tests might cause errors irrelevant to your changes.
   In such a case, we ignore the failures and go on to the review process, so do not worry about it!


Documentation
-------------

When adding a new feature to the framework, you also need to document it in the reference.
For example, if you are adding a new function under ``chainer.functions``, you need to add it to the :doc:`reference/functions` page.

.. note::

   If you are unsure about how to fix the documentation, you can submit a pull request without doing so.
   Reviewers will help you fix the documentation appropriately.

The documentation source is stored under `docs directory <https://github.com/chainer/chainer/tree/master/docs>`_ and written in `reStructuredText <http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ format.

To build the documentation, you need to install `Sphinx <http://www.sphinx-doc.org/>`_::

  $ pip install sphinx sphinx_rtd_theme

Then you can build the documentation in HTML format locally::

  $ cd docs
  $ make html

HTML files are generated under ``build/html`` directory.
Open ``index.html`` with the browser and see if it is rendered as expected.

.. note::

   Docstrings (documentation comments in the source code) are collected from the installed Chainer module.
   If you modified docstrings, make sure to install the module (e.g., using `pip install -e .`) before building the documentation.
