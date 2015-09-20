import sys
import unittest

import numpy

import chainer


class Object1(chainer.ParameterizedObject):

    def __init__(self):
        super(Object1, self).__init__()
        self.params['w'] = chainer.Variable(numpy.array([1, 2, 3], dtype='f'))
        self.states['s'] = numpy.array([4, 5, 6], dtype='f')


class TestParameterizedObject(unittest.TestCase):

    def setUp(self):
        self.obj = Object1()

    def test_name(self):
        self.assertEqual(self.obj.name, '/')
        self.obj.name = '/foo'
        self.assertEqual(self.obj.name, '/foo')

    def test_copy_shared(self):
        src = self.obj
        src.name = '/foo'
        dst = src.copy()

        self.assertEqual(dst.name, src.name)
        self.assertIsNot(dst.params, src.params)
        self.assertIsNot(dst.states, src.states)
        self.assertIsNot(dst.params['w'], src.params['w'])
        self.assertIs(dst.params['w'].data, src.params['w'].data)
        self.assertIs(dst.states['s'], src.states['s'])

    def test_copy_not_shared(self):
        src = self.obj
        src.name = '/foo'
        dst = src.copy(shared=False)

        self.assertEqual(dst.name, src.name)
        numpy.testing.assert_array_equal(
            dst.params['w'].data, src.params['w'].data)
        self.assertIsNot(dst.params['w'].data, src.params['w'].data)
        numpy.testing.assert_array_equal(
            dst.states['s'], src.states['s'])
        self.assertIsNot(dst.states['s'], src.states['s'])

    def test_visitparams(self):
        obj = self.obj
        obj.name = '/foo'
        w = obj.params['w']
        w.grad = numpy.empty_like(w.data)
        params = tuple(obj.visitparams())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0][0], '/foo/_params/w')
        self.assertIs(params[0][1], w)

    def test_visithierarchy(self):
        obj = self.obj
        objs = tuple(obj.visithierarchy())
        self.assertEqual(len(objs), 1)
        self.assertIs(objs[0], obj)

    def test_copyparams(self):
        src = Object1()
        dst = self.obj
        w = chainer.Variable(numpy.ndarray(3, dtype='f'))
        src.params['w'] = w
        w_dst = dst.params['w']

        dst.copyparams(src)
        numpy.testing.assert_array_equal(w_dst.data, w.data)
        self.assertIs(dst.params['w'].data, w_dst.data)

    def test_addgrads(self):
        g_src = numpy.array([1, 2, 3], dtype='f')
        g_dst = numpy.array([2, 3, 4], dtype='f')

        src = Object1()
        src.params['w'].grad = g_src.copy()
        self.obj.params['w'].grad = g_dst.copy()
        self.obj.addgrads(src)
        numpy.testing.assert_array_equal(
            self.obj.params['w'].grad, g_src + g_dst)

    def test_zerograds(self):
        obj = self.obj
        self.assertIs(obj.params['w'].grad, None)
        obj.zerograds()
        numpy.testing.assert_array_equal(
            obj.params['w'].grad,
            numpy.zeros_like(obj.params['w'].data))


class TestParameterizedDict(unittest.TestCase):

    def setUp(self):
        self.obj = chainer.ParameterizedDict(
            ch1=Object1(),
            ch2=Object1(),
        )

    def test_elem_name(self):
        self.assertEqual(self.obj['ch1'].name, '/ch1')
        self.assertEqual(self.obj['ch2'].name, '/ch2')

    def test_init_non_model(self):
        self.assertRaises(TypeError, chainer.ParameterizedDict, o1=0)

    def test_init_multiple_parents(self):
        o = Object1()
        o.name = '/foo'
        self.assertRaises(ValueError, chainer.ParameterizedDict, bar=o)
        self.assertRaises(ValueError, chainer.ParameterizedDict,
                          bar=self.obj['ch1'])

    def test_contains(self):
        self.assertIn('ch1', self.obj)
        self.assertIn('ch2', self.obj)
        self.assertNotIn('ch3', self.obj)
        self.assertNotIn('_params', self.obj)

    def test_delitem(self):
        model = self.obj['ch1']
        del self.obj['ch1']
        self.assertNotIn('ch1', self.obj)
        self.assertEqual(model.name, '/')

    def test_iter(self):
        d = dict(self.obj)
        self.assertIn('ch1', d)
        self.assertIs(d['ch1'], self.obj['ch1'])
        self.assertIn('ch2', d)
        self.assertIs(d['ch2'], self.obj['ch2'])

    def test_setitem(self):
        old_obj = self.obj['ch1']
        new_obj = Object1()
        self.obj['ch1'] = new_obj
        self.assertIs(self.obj['ch1'], new_obj)
        self.assertEqual(new_obj.name, '/ch1')
        self.assertEqual(old_obj.name, '/')

    def test_setitem_non_model(self):
        self.assertRaises(TypeError, self.obj.__setitem__, 0)

    def test_setitem_multiple_parents(self):
        o = Object1()
        o.name = '/foo'
        self.assertRaises(ValueError, self.obj.__setitem__, 'bar', o)

    def test_clear(self):
        ch1 = self.obj['ch1']
        ch2 = self.obj['ch2']
        self.obj.clear()
        self.assertNotIn('ch1', self.obj)
        self.assertNotIn('ch2', self.obj)
        self.assertEqual(ch1.name, '/')
        self.assertEqual(ch2.name, '/')

    def test_get(self):
        self.assertIs(self.obj.get('ch1'), self.obj['ch1'])
        self.assertIs(self.obj.get('ch3', None), None)

    def test_items(self):
        items = dict(self.obj.items())
        self.assertIn('ch1', items)
        self.assertIs(items['ch1'], self.obj['ch1'])
        self.assertIn('ch2', items)
        self.assertIs(items['ch2'], self.obj['ch2'])

    if sys.version_info.major < 3:
        def test_iteritems(self):
            items = dict(self.obj.iteritems())
            self.assertIn('ch1', items)
            self.assertIs(items['ch1'], self.obj['ch1'])
            self.assertIn('ch2', items)
            self.assertIs(items['ch2'], self.obj['ch2'])

        def test_iterkeys(self):
            keys = list(self.obj.iterkeys())
            self.assertEqual(len(keys), 2)
            self.assertIn('ch1', keys)
            self.assertIn('ch2', keys)

        def test_itervalues(self):
            values = list(self.obj.itervalues())
            self.assertEqual(len(values), 2)
            self.assertIn(self.obj['ch1'], values)
            self.assertIn(self.obj['ch2'], values)

    def test_has_key(self):
        self.assertTrue(self.obj.has_key('ch1'))
        self.assertTrue(self.obj.has_key('ch2'))
        self.assertFalse(self.obj.has_key('ch3'))

    def test_keys(self):
        keys = list(self.obj.keys())
        self.assertEqual(len(keys), 2)
        self.assertIn('ch1', keys)
        self.assertIn('ch2', keys)

    def test_pop(self):
        ch1 = self.obj.pop('ch1')
        self.assertEqual(len(self.obj), 1)
        self.assertNotIn('ch1', self.obj)
        self.assertIn('ch2', self.obj)
        self.assertEqual(ch1.name, '/')

        self.assertRaises(KeyError, self.obj.pop, 'ch1')
        self.assertEqual(self.obj.pop('ch1', 100), 100)

    def test_popitem(self):
        d = dict(self.obj)
        k, o = self.obj.popitem()
        self.assertIn(k, ('ch1', 'ch2'))
        self.assertIs(o, d[k])
        self.assertEqual(o.name, '/')

    def test_setdefault(self):
        ch1 = self.obj['ch1']
        self.assertIs(self.obj.setdefault('ch1'), ch1)
        self.assertIs(self.obj['ch1'], ch1)

        o = Object1()
        ret = self.obj.setdefault('ch3', o)
        self.assertIn('ch3', self.obj)
        self.assertIs(self.obj['ch3'], o)
        self.assertEqual(o.name, '/ch3')

        self.assertRaises(TypeError, self.obj.setdefault, 'ch4')

        o2 = Object1()
        o2.name = '/foo'
        self.assertRaises(ValueError, self.obj.setdefault, 'ch4', o2)

    def test_values(self):
        values = list(self.obj.values())
        self.assertEqual(len(values), 2)
        self.assertIn(self.obj['ch1'], values)
        self.assertIn(self.obj['ch2'], values)

    def test_name(self):
        self.obj.name = '/foo'
        self.assertEqual(self.obj.name, '/foo')
        self.assertEqual(self.obj['ch1'].name, '/foo/ch1')
        self.assertEqual(self.obj['ch2'].name, '/foo/ch2')

    def test_visithierarchy(self):
        objs = list(self.obj.visithierarchy())
        self.assertEqual(len(objs), 3)
        self.assertIs(objs[0], self.obj)
        self.assertIn(objs[1], self.obj.values())
        self.assertIn(objs[2], self.obj.values())


class TestParameterizedList(unittest.TestCase):

    def setUp(self):
        self.obj = chainer.ParameterizedList(Object1(), Object1())

    def test_elem_name(self):
        self.assertEqual(self.obj[0].name, '/0')
        self.assertEqual(self.obj[1].name, '/1')

    def test_init_non_model(self):
        self.assertRaises(TypeError, chainer.ParameterizedList, 0)

    def test_init_multiple_parents(self):
        o = Object1()
        o.name = '/foo'
        self.assertRaises(ValueError, chainer.ParameterizedList, o)

    def test_getitem(self):
        o0 = Object1()
        o1 = Object1()
        l = chainer.ParameterizedList(o0, o1)
        self.assertIs(l[0], o0)
        self.assertIs(l[1], o1)
        self.assertRaises(IndexError, l.__getitem__, 2)

    def test_iter(self):
        l = list(self.obj)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], self.obj[0])
        self.assertIs(l[1], self.obj[1])

    def test_len(self):
        self.assertEqual(len(self.obj), 2)

    def test_setitem(self):
        old = self.obj[0]
        new = Object1()
        self.obj[0] = new
        self.assertIs(self.obj[0], new)
        self.assertEqual(new.name, '/0')
        self.assertEqual(old.name, '/')

    def test_setitem_non_model(self):
        self.assertRaises(TypeError, self.obj.__setitem__, 0, 0)

    def test_setitem_multiple_parents(self):
        o = Object1()
        o.name = '/foo'
        self.assertRaises(ValueError, self.obj.__setitem__, 0, o)

    def test_setitem_out_of_index(self):
        self.assertRaises(IndexError, self.obj.__setitem__, 2, Object1())

    def test_append(self):
        o = Object1()
        self.obj.append(o)
        self.assertEqual(len(self.obj), 3)
        self.assertIs(self.obj[2], o)
        self.assertEqual(o.name, '/2')

    def test_append_non_model(self):
        self.assertRaises(TypeError, self.obj.append, 0)

    def test_append_multiple_parents(self):
        o = Object1()
        o.name = '/foo'
        self.assertRaises(ValueError, self.obj.append, o)

    def test_pop(self):
        o0 = self.obj[0]
        o1 = self.obj[1]
        p = self.obj.pop()
        self.assertIs(o1, p)
        self.assertEqual(len(self.obj), 1)
        self.assertIs(self.obj[0], o0)
        self.assertEqual(p.name, '/')

    def test_name(self):
        self.obj.name = '/foo'
        self.assertEqual(self.obj[0].name, '/foo/0')
        self.assertEqual(self.obj[1].name, '/foo/1')

    def test_visithierarchy(self):
        objs = list(self.obj.visithierarchy())
        self.assertEqual(len(objs), 3)
        self.assertIs(objs[0], self.obj)
        self.assertIs(objs[1], self.obj[0])
        self.assertIs(objs[2], self.obj[1])
