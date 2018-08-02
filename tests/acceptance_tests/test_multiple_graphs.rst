Acceptance tests for multiple graphs
====================================

Double backprop with different graphs
-------------------------------------

>>> import xchainer as xc

>>> with xc.graph_scope('weight') as weight_graph:
...     with xc.graph_scope('input') as input_graph:
...         x = xc.ndarray((3,), xc.float32, [1, 2, 3]).require_grad(input_graph)
...         w = xc.ndarray((3,), xc.float32, [4, 5, 6]).require_grad(weight_graph)
...         y = x * w
...         y.is_grad_required(input_graph)
True

...         y.is_grad_required(weight_graph)
True

...         y.is_grad_required()  # 'default'
False

...         xc.backward(y, graph_id=input_graph)
...         gx = x.get_grad(input_graph)
...         gx  # == w
array([4., 5., 6.], shape=(3,), dtype=float32, device='native:0', graph_ids=['weight'])

...         w.get_grad(input_graph)
Traceback (most recent call last):
  ...
xchainer.XchainerError: Array does not belong to the graph: 'input'.

...     z = gx * w  # == w * w
...     xc.backward(z, graph_id=weight_graph)
...     w.get_grad(weight_graph)  # == 2 * w
array([ 8., 10., 12.], shape=(3,), dtype=float32, device='native:0')

...     x.get_grad(weight_graph)
Traceback (most recent call last):
  ...
xchainer.XchainerError: Array does not belong to the graph: 'weight'.

Double backprop with single graph
---------------------------------

>>> x = xc.ndarray((3,), xc.float32, [1, 2, 3]).require_grad()
>>> w = xc.ndarray((3,), xc.float32, [4, 5, 6]).require_grad()
>>> y = x * w
>>> y.is_grad_required()
True
>>> with xc.graph_scope('foo') as foo:
...     y.is_grad_required(foo)  # unknown graph name
False

>>> xc.backward(y, enable_double_backprop=True)
>>> gx = x.get_grad()
>>> gx  # == w
array([4., 5., 6.], shape=(3,), dtype=float32, device='native:0', graph_ids=['<default>'])
>>> w.get_grad()  # == x
array([1., 2., 3.], shape=(3,), dtype=float32, device='native:0', graph_ids=['<default>'])

>>> w.cleargrad()
>>> z = gx * w  # == w * w
>>> xc.backward(z)
>>> w.get_grad()  # == 2 * w
array([ 8., 10., 12.], shape=(3,), dtype=float32, device='native:0')
>>> x.get_grad()  # the second backprop does not reach here
array([4., 5., 6.], shape=(3,), dtype=float32, device='native:0', graph_ids=['<default>'])
>>> x.get_grad() is gx
True
