.. _contrib:

Contribution Guide
==================

Chainer is an `open source software hosted on GitHub <https://github.com/chainer/chainer>`_ and welcomes contributors to take part in the development of the framework.
This is a document aimed towards such contributors.
Anyone who for instance would like to file an issue or send a pull request (PR) is encouraged to go through it.

Issues and Pull Requests
------------------------

First steps in contributing to Chainer often involve filing an issue or creating a PR.
This section describes how to do so.

How to File an Issue
~~~~~~~~~~~~~~~~~~~~

To file an issue on GitHub, you often only need to follow instructions given by the template.
Write precise explanations on how you want Chainer to behave or include necessary and sufficient conditions to reproduce the bugs.
Feature requests should include **what** you want to do and preferably **why**.
You may additionally suggest **how**.

.. warning::

   If you have a question regarding the usage of Chainer, it is recommended that you send a post to `StackOverflow <https://stackoverflow.com/>`_ or the `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_ instead of the issue tracker.
   The issue tracker is not a place to share knowledge on practices.

How to Send a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you can write code to fix an issue, it is encouraged to send a PR.

In that case, confirm the following points before starting to write any code.

- Read :ref:`coding-guide` and :ref:`testing-guide`.
- Check the appropriate branch to which you should send a PR, following :ref:`contrib-git-branches`.
  If you are unsure about which branch to target, choose the ``master`` branch.
  The current source tree of the chosen branch is the starting point of your change.

After writing your code **(including unit tests and hopefully documentations!)**, send a PR on GitHub.
You have to write a precise explanation of **what** and **how** in the description;
this is the first documentation of your code and an important part of your PR.

However, even if your code is not complete, you can send a PR as a *work-in-progress (WIP) PR* by prefixing the PR title with ``[WIP]``.
If you just describe the PR, the core team and other contributors can join the discussion about how to proceed with it.
WIP PRs may occasionally be useful for discussing based on concrete code.

When a PR is created (or updated), it is automatically tested in one of our CI environments, namely Travis CI.
There are other CI environments as well often manually triggered by the reviewer.
The various CIs are required to test for instance different platforms or CUDA environments.
Once the tests in all CI environments pass and/or the PR is approved by the reviewer, the PR will be merged.

.. note::

    If you are planning to add a new feature or modify existing APIs, **it is recommended that you open an issue and discuss the design first.**
    Following the consequences of the discussions, you can send a PR that is smoothly reviewed in a shorter time.

Issue/Pull Request Labels
~~~~~~~~~~~~~~~~~~~~~~~~~

Issues and PRs are labeled on GitHub so that they can be grouped, filtered and better maintained.
For instance, a label can indicate that a ticket needs response from the PR author, or that an issue needs immediate action in case of a critical bug.
Please refer to the `list of lables on GitHub <https://github.com/chainer/chainer/labels>`_.

.. _coding-guide:

Coding Guidelines
-----------------

We follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and partially `OpenStack Style Guidelines <https://docs.openstack.org/developer/hacking/>`_ as basic style guidelines.
Any contributions in terms of code are expected to follow these guidelines.

You can use the ``autopep8`` and the ``flake8`` commands to check whether or not your code follows the guidelines.
In order to avoid confusion from using different tool versions, we pin the versions of those tools.
Install them with the following command (from within the top directory of the Chainer repository)::

  $ pip install -e '.[stylecheck]'

And check your code with::

  $ autopep8 path/to/your/code.py
  $ flake8 path/to/your/code.py

``autopep8`` can automatically correct Python code to conform to the PEP 8 style guide::

  $ autopep8 --in-place path/to/your/code.py

The ``flake8`` command lets you know parts of your code that are not following the style guidelines.

Note that ``flake8`` command is not perfect.
It does not check some of the style guidelines.
Here is a (not-exhaustive) list of the rules that ``flake8`` cannot check.

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
   Use of such aliases was prohibited in the past for avoiding confusing errors related to cyclic dependencies;
   we relaxed the rule so that the library code looks similar to user code.

   When you use such shortcut aliases, please be careful of cyclic imports.
   One of the typical pitfalls is a way to import ``chainer.functions``.
   An import like ``import chainer.functions as F`` within modules under ``chainer.functions`` does not work.
   An import like ``from chainer import functions`` works well with Python 3, but does not with Python 2.
   We recommend you to use ``import chainer.functions`` and spell out like ``chainer.functions.foo`` in your methods.

.. _testing-guide:

Unit Testing
------------

Testing is one of the most important aspects of your PR.
You should write test cases and verify your implementation by following the testing guide above.
If you modify code related to existing unit tests, you must run appropriate commands and confirm that the tests still pass.

Note that we are using ``pytest`` and the ``mock`` package for testing.
They are not included in Chainer and need to be installed as follows::

  $ pip install pytest mock

How to Run Tests
~~~~~~~~~~~~~~~~

You can run all unit tests with the following command from the root directory of the Chainer::

  $ python -m pytest

Or specify a test script that you want to run::

  $ python -m pytest path/to/your/test.py

You can also run all unit tests under a specific directory::

  $ python -m pytest tests/chainer_tests/<directory name>

Some tests require CUDA and cuDNN by default.
In order to run unit tests that do not require CUDA and cuDNN, set an environment variable and filter using test marks as follows::

  $ export CHAINER_TEST_GPU_LIMIT=0
  $ python -m pytest path/to/your/test.py -m='not cudnn'

Some GPU tests involve multiple GPUs.
If you want to run GPU tests with insufficient number of GPUs, specify the number of available GPUs to ``CHAINER_TEST_GPU_LIMIT``.
For example, if you only have a single GPU, launch ``pytest`` with the following command to skip multi-GPU tests::

  $ export CHAINER_TEST_GPU_LIMIT=1
  $ python -m pytest path/to/gpu/test.py

Some tests spend too much time.
If you want to skip such tests, pass ``-m='not slow'`` option to the command::

  $ python -m pytest path/to/your/test.py -m='not slow'

Test File and Directory Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests are found in the :tree:`tests/chainer_tests` directory.
In order to enable the test runner to find test scripts correctly, we are using a special naming convention for the test subdirectories and the test scripts.

* The name of each subdirectory of ``tests`` must end with the ``_tests`` suffix.
* The name of each test script must start with the ``test_`` prefix.

When we write a test for a module, we use the appropriate path and file name for the test script whose correspondence to the tested module is clear.
For example, if you want to write a test for a module ``chainer.x.y.z``, the test script must be located at ``tests/chainer_tests/x_tests/y_tests/test_z.py``.

How to Write Tests
~~~~~~~~~~~~~~~~~~

There are many examples of unit tests under the :tree:`tests` directory, so reading some of them is a good and recommended way to learn how to write tests for Chainer.
They use the :mod:`unittest` package of the standard library, while some tests are additionally using utilities from :mod:`chainer.testing`.

In addition to the :ref:`coding-guide` mentioned above, the following rules apply to the test code:

* All test classes must inherit from :class:`unittest.TestCase`.
* Use :mod:`unittest` features to write tests, except for the following cases:

    * Use ``assert`` statement instead of ``self.assert*`` methods (e.g., write ``assert x == 1`` instead of ``self.assertEqual(x, 1)``).
    * Use ``with pytest.raises(...):`` instead of ``with self.assertRaises(...):``.

.. note::

   We are incrementally applying the above style.
   Some existing tests may be using the old style (``self.assertRaises``, etc.), but all newly written tests should follow the above style.

Even if your patch includes GPU-related code, your tests should not fail without GPU capability.
Test functions that require CUDA must be tagged with the ``chainer.testing.attr.gpu`` decorator::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.gpu
      def test_my_gpu_func(self):
          ...

The functions tagged with the ``gpu`` decorator are skipped if ``CHAINER_TEST_GPU_LIMIT=0`` environment variable is set.
We also have the ``chainer.testing.attr.cudnn`` decorator to let ``pytest`` know that the test depends on cuDNN.
The test functions decorated with ``cudnn`` are skipped if ``-m='not cudnn'`` is given.

The test functions decorated with ``gpu`` must not depend on multiple GPUs.
In order to write tests for multiple GPUs, use the ``chainer.testing.attr.multi_gpu()`` decorator instead::

  import unittest
  from chainer.testing import attr

  class TestMyFunc(unittest.TestCase):
      ...

      @attr.multi_gpu(2)  # specify the number of required GPUs here
      def test_my_two_gpu_func(self):
          ...

If your test requires too much time, add the ``chainer.testing.attr.slow`` decorator.
The test functions decorated with ``slow`` are skipped if ``-m='not slow'`` is given::

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

Documentation
-------------

When adding a new feature to the framework, you should also document it in the reference so that other users can find it in the official documentation.
For example, if you are adding a new function under ``chainer.functions``, :doc:`reference/functions` should be updated.

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

.. note::

   If you are unsure about how to write the documentation or failed to build it locally, you can submit a PR without documentation.
   Reviewers will help you with it.

Other Forms of Contribution
---------------------------

There are several other ways in which you can contribute to Chainer without directly working with the code base.
Following are such contributions.

* Sending a question/reply to `StackOverflow <https://stackoverflow.com/>`_ (with ``chainer`` tag) or `Chainer User Group <https://groups.google.com/forum/#!forum/chainer>`_
* Open-sourcing an external example
* Writing a post about Chainer

Development Cycle
-----------------

This section explains the development process of Chainer.

Versioning
~~~~~~~~~~

The versioning of Chainer follows `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ and a part of `Semantic versioning <https://semver.org/>`_.
The version number consists of three or four parts: ``X.Y.Zw`` where ``X`` denotes the **major version**, ``Y`` denotes the **minor version**, ``Z`` denotes the **revision number**, and the optional ``w`` denotes the pre-release suffix.
While the major, minor, and revision numbers follow the rule of semantic versioning, the pre-release suffix follows PEP 440, the Python community standards.

**Note that a major update basically does not contain compatibility-breaking changes from the last release candidate (RC).**
This is not a strict rule, though; if there is a critical bug in the API that need to be fixed for the major version, breaking changes may be introduced.

For more on backward compatibility, please refer to the :ref:`compatibility`.

.. _contrib-release-cycle:

Release Cycle
~~~~~~~~~~~~~

Two tracks with different versions are developed in parallel.
The first track is the **stable versions**, which is a series of minor (occasional revision) updates for the latest major version.
The second track is the **development versions**, which is a series of pre-releases for the upcoming major version.

If ``X.0.0`` is the latest major version, followed by ``Y.0.0`` and ``Z.0.0``,  a typical release cycle timeline would be as follows.

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
In particular, each revision/minor release is made around four weeks after the previous one of the same major version, and the pre-release of the upcoming major version is made at the same time.
Whether these releases are revision or minor is determined based on the contents of each update.

Note that there are only three stable releases for the versions ``X.x.x`` in the example table above.
The number of stable releases may vary depending the development status of the following in this case ``Y`` and its number of required beta versions (a ``b`` followed by a number).
During the parallel development of ``Y.0.0`` and ``Z.0.0a1``, the version ``Y`` is treated as an **almost-stable version** and ``Z`` is treated as a development version.

If there is a critical bug found in ``X.x.x`` after stopping the development of version ``X``, we may release a hot-fix for this version at any time.

A `milestone for each upcoming release is published on GitHub <https://github.com/chainer/chainer/milestones>`_.
The GitHub milestones are used to group issues and PRs belonging to a release.

.. _contrib-git-branches:

Git Branches
~~~~~~~~~~~~

The ``master`` branch is used to develop pre-release versions.
It means that **alpha, beta, and RC updates are developed at the** ``master`` **branch**.
This branch contains the most up-to-date source tree that includes features newly added after the latest major version.

The stable version is developed on the ``vN`` branch where "N" reflects the version number (*versioned branch*).
For example, v3.0.0, v3.1.0, and v3.2.0 are developed on the ``v3`` branch.

A PR from a contributor should in general be targeting the ``master`` branch.
If the change can and should be applied to the stable version in addition, a member from the core team will make sure it is backported to be included in the next revision update.

If the change is only applicable to the stable version and not to the ``master`` branch, please send it to the versioned branch.
We basically only accept changes to the latest versioned branch (where the stable version is developed) unless the fix is critical.

If you want to introduce a new feature in the ``master`` branch to the current stable version, please send a *backport PR* to the stable version (the latest ``vN`` branch).
See the next section for details.

*Note: a change that can be applied to both branches should be sent to the* ``master`` *branch.*

Feature Backport Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general, new features in the development branch are not backported to the stable versions.
If such backports can be motivated and are necessary however, they are welcomed.
In such a case, a backport PR must be sent to the latest ``vN`` branch.
**Note that we do not accept any feature backport PRs to older versions because we are not running quality assurance workflows (e.g. CI) for older versions which means that we cannot ensure that the PR is correctly ported.**

There are some rules on sending backport PRs.

* Prefix the PR title with **[backport]**.
* Include the original PR number in the PR description, e.g. "This is a backport of #XXXX".
* (Optional) Write in the PR description, the motivation behind the backport.

There is a `backport tool <https://github.com/chainer/backport>`_ maintained by the core team that automates the process of creating backport PRs conforming to the rules above.

Note: PRs that do not include any changes/additions to APIs (e.g. bug fixes, documentation improvements) are backported by the core team, but contributors are also welcomed to do so to make development faster.
