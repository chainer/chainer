{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   ..
      Methods

{% block methods %}

   .. rubric:: Methods

   ..
      Special methods

{% for item in ('__call__', '__enter__', '__exit__', '__getitem__', '__setitem__', '__len__', '__next__', '__iter__', '__copy__') %}
{% if item in all_methods %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Ordinary methods

{% for item in methods %}
{% if item not in ('__init__',) %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Special methods

{% for item in ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__', '__nonzero__', '__bool__') %}
{% if item in all_methods %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Special (arithmetic) methods

{% for item in ('__neg__', '__abs__', '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__div__', '__truediv__', '__rdiv__', '__rtruediv__', '__floordiv__', '__rfloordiv__', '__pow__', '__rpow__', '__matmul__', '__rmatmul__') %}
{% if item in all_methods %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

{% endblock %}

   ..
      Attributes

{% block attributes %} {% if attributes %}

   .. rubric:: Attributes

{% for item in attributes %}
   .. autoattribute:: {{ item }}
{%- endfor %}
{% endif %} {% endblock %}
