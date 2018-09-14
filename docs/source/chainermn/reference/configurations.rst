Environmental Variables
=======================

``CHAINERMN_FORCE_ABORT_ON_EXCEPTIONS``
  If this variable is set to a non-empty value,
  ChainerMN installs a global hook to Python's `sys.excepthook` to call ``MPI_Abort()`` when
  an unhandled exception occurs. See :ref:`faq-global-except-hook`

  `ChainerMN issue #236 <https://github.com/chainer/chainermn/issues/236>`_
  may also help to understand the problem.
  

Execution Control
=================

.. autofunction:: chainermn.global_except_hook.add_hook
