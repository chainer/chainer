Acceptance tests for multiple graphs
====================================

Double backprop with different graphs
-------------------------------------

>>> import xchainer as xc

>>> x = xc.Array((3,), xc.float32, [1, 2, 3]).require_grad('input')
>>> w = xc.Array((3,), xc.float32, [4, 5, 6]).require_grad('weight')
>>> y = x * w
>>> y.is_grad_required('input')
True
>>> y.is_grad_required('weight')
True
>>> y.is_grad_required()  # 'default'
False

>>> xc.backward(y, graph_id='input')
>>> gx = x.get_grad('input')
>>> gx  # == w
array([4., 5., 6.], dtype=float32, graph_ids=['weight'])
>>> w.get_grad('input')
Traceback (most recent call last):
  ...
xchainer.XchainerError: Cannot find ArrayNode for graph: input

>>> z = gx * w  # == w * w
>>> xc.backward(z, graph_id='weight')
>>> w.get_grad('weight')  # == 2 * w
array([ 8., 10., 12.], dtype=float32)
>>> x.get_grad('weight')
Traceback (most recent call last):
  ...
xchainer.XchainerError: Cannot find ArrayNode for graph: weight


Double backprop with single graph
---------------------------------

>>> x = xc.Array((3,), xc.float32, [1, 2, 3]).require_grad()
>>> w = xc.Array((3,), xc.float32, [4, 5, 6]).require_grad()
>>> y = x * w
>>> y.is_grad_required()
True
>>> y.is_grad_required('foo')  # unknown graph name
False

>>> xc.backward(y, enable_double_backprop=True)
>>> gx = x.get_grad()
>>> gx  # == w
array([4., 5., 6.], dtype=float32, graph_ids=['default'])
>>> w.get_grad()  # == x
array([1., 2., 3.], dtype=float32, graph_ids=['default'])

>>> w.clear_grad()
>>> z = gx * w  # == w * w
>>> xc.backward(z)
>>> w.get_grad()  # == 2 * w
array([ 8., 10., 12.], dtype=float32)
>>> x.get_grad()  # the second backprop does not reach here
array([4., 5., 6.], dtype=float32, graph_ids=['default'])
>>> x.get_grad() is gx
True
