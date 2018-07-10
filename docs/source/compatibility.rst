.. _compatibility:

API Compatibility Policy
========================

This documentation explains the design policy on compatibilities of Chainer APIs.
Development team should follow this policy on deciding to add, extend, and change APIs and their behaviors.

This documentation is written for both users and developers.
Users can decide the level of dependencies on Chainer’s implementations in their codes based on this document.
Developers should read through this documentation before creating pull requests that contain changes on the interface.
Note that this documentation may contain ambiguities on the level of supported compatibilities.


Versioning and Backward Compatibility
-------------------------------------

The versioning of Chainer follows the `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ and a part of `Semantic versioning <https://semver.org/>`_.
See :ref:`contrib` for details of versioning.

The backward compatibility is kept for **revision updates** and **minor updates**, which are applied to the stable version.
A **major update** from the latest release candidate basically keeps the backward compatibility, although it is not guaranteed.
Any **pre-releases** may break the backward compatibility.


Breaking the Compatibility
--------------------------

We sometimes need to break the backward compatibility to improve the framework design and to support new kinds of machine learning methods.
Such a change is only made into pre-releases (alpha, beta, and release candidate) and sometimes into the major update.

A change that breaks the compatibility affects user codes.
We try to lower the cost of adapting your code to the newer version.
The following list shows an example of what we can do to reduce the cost (*Note: this is not a promise; what kind of actions we can take depends on the situation*).

- When an argument is removed from an existing API, passing the argument to the updated API will emit an error with a special error message.
  The error message tells you how to fix your code.
- When a function or a class is removed, we make the current stable version emit a deprecation warning.
  **Note that the deprecation warning is not printed by default in Python.**
  You have to manually turn on the deprecation warning by ``warnings.simplefilter('always', DeprecationWarning)``.
- When a definition of a link is changed, we try to enable it to deserialize a model dumped with an older version of Chainer.
  In most cases, we cannot guarantee that a model serialized with a newer version of Chainer is loadable by an older version of Chainer.

.. note::

   Since Chainer v2, we have stopped adopting any solid processes to break backward compatibilities (e.g. a solid schedule for deprecating and removing a feature) in order to keep the development fast enough to support the cutting-edge research.
   **It does not mean we stop taking care of maintainability of user codes.**
   We are still paying much attention to not breaking user codes.


.. module:: chainer.utils

Experimental APIs
-----------------

Thanks to many contributors, we have introduced many new features to Chainer.

However, we have sometimes released new features only to later notice that their APIs are not appropriate.
In particular, we sometimes know that the API is likely to be modified in the near future because we do not have enough knowledge about how well the current design fits to the real usages.
**The objective of experimental APIs is to declare that the APIs are likely to be updated in the near future so that users can decide if they can(not) use them.**

Any newly added API can be marked as *experimental*.
Any API that is not experimental is called *stable* in this document.

.. note::

    Undocumented behaviors are not considered as APIs, so they can be changed at any time (even in a revision update).
    The treatment of undocumented behaviors are described in :ref:`undocumented_behavior` section.

When users use experimental APIs for the first time, warnings are raised once for each experimental API,
unless users explicitly disable the emission of the warnings in advance.

See the documentation of :meth:`chainer.utils.experimental` to know how developers mark APIs as experimental
and how users enable or disable the warnings practically.

.. note::

   It is up to developers if APIs should be annotated as experimental or not.
   We recommend to make the APIs experimental if they implement large modules or
   make a decision from several design choices.


Supported Backward Compatibility
--------------------------------

This section defines backward compatibilities that revision updates must maintain.

Documented Interface
~~~~~~~~~~~~~~~~~~~~

Chainer has the official API documentation.
Many applications can be written based on the documented features.
We support backward compatibilities of documented features.
In other words, codes only based on the documented features run correctly with revision-updated versions.

Developers are encouraged to use apparent names for objects of implementation details.
For example, attributes outside of the documented APIs should have one or more underscores at the prefix of their names.

.. note::

   Although it is not stated as a rule, we also try to keep the compatibility for any interface that looks like a stable feature.
   For example, if the name of a symbol (function, class, method, attribute, etc.) is not prefixed by an underscore and the API is not experimental,
   the API should be kept over revision updates even if it is not documented.

.. _undocumented_behavior:

Undocumented behaviors
~~~~~~~~~~~~~~~~~~~~~~

Behaviors of Chainer implementation not stated in the documentation are undefined.
Undocumented behaviors are not guaranteed to be stable between different revision versions.

Even revision updates may contain changes to undefined behaviors.
One of the typical examples is a bug fix.
Another example is an improvement on implementation, which may change the internal object structures not shown in the documentation.
As a consequence, **even revision updates do not support compatibility of pickling, unless the full layout of pickled objects is clearly documented.**

Documentation Error
~~~~~~~~~~~~~~~~~~~

Compatibility is basically determined based on the documentation, although it sometimes contains errors.
It may make the APIs confusing to assume the documentation always stronger than the implementations.
We therefore may fix the documentation errors in any updates that may break the compatibility in regard to the documentation.

.. note::

   Developers should not fix the documentation and implementation of the same functionality at the same time in revision updates as a "bug fix"
   unless the bug is so critical that no users are expected to be using the old version correctly.

Object Attributes and Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Object attributes and properties are sometimes replaced by each other.
It does not break the user codes, except the codes depend on how the attributes and properties are implemented.

Functions and Methods
~~~~~~~~~~~~~~~~~~~~~

Methods may be replaced by callable attributes keeping the compatibility of parameters and return values.
It does not break the user codes, except the codes depend on how the methods and callable attributes are implemented.

Exceptions and Warnings
~~~~~~~~~~~~~~~~~~~~~~~

The specifications of raising exceptions are considered as a part of standard backward compatibilities.
No exception is raised in the future revision versions with correct usages that the documentation allows.

On the other hand, warnings may be added at any revision updates for any APIs.
It means revision updates do not keep backward compatibility of warnings.

Model Format Compatibility
--------------------------

Links and chains serialized by official serializers that Chainer provides are correctly loaded with the future versions.
They might not be correctly loaded with Chainer of the lower versions.

.. note::

   Current serialization APIs do not support versioning.
   It prevents us from introducing changes in the layout of objects that support serialization.
   We are discussing versioning in serialization APIs.

Installation Compatibility
--------------------------

The installation process is another concern of compatibilities.

Any changes on the set of dependent libraries that force modifications on the existing environments should be done in pre-releases and major updates.
Such changes include following cases:

- dropping supported versions of dependent libraries (e.g. dropping cuDNN v2)
- adding new mandatory dependencies (e.g. adding h5py to setup_requires)

.. note::

   We sometimes have to narrow the supported versions due to bugs in the specific versions of libraries.
   In such a case, we may drop the support of those versions even in revision updates unless a workaround is found for the issue.
