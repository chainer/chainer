import sys
import unittest

import numpy

import chainer


class Model1(chainer.Model):

    def __init__(self):
        chainer.Model.__init__(self)
        self.params['w'] = numpy.array([1, 2, 3], dtype='f')
        self.states['s'] = numpy.array([4, 5, 6], dtype='f')


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model1()

    def test_name(self):
        self.assertEqual(self.model.name, '/')
        self.model.name = '/foo'
        self.assertEqual(self.model.name, '/foo')

    def test_copy_shared(self):
        src = self.model
        src.name = '/foo'
        dst = src.copy()

        self.assertEqual(dst.name, src.name)
        self.assertIsNot(dst.params, src.params)
        self.assertIsNot(dst.states, src.states)
        self.assertIs(dst.params['w'], src.params['w'])
        self.assertIs(dst.states['s'], src.states['s'])

    def test_copy_not_shared(self):
        src = self.model
        src.name = '/foo'
        dst = src.copy(shared=False)

        self.assertEqual(dst.name, src.name)
        numpy.testing.assert_array_equal(
            dst.params['w'], src.params['w'])
        self.assertIsNot(dst.params['w'], src.params['w'])
        numpy.testing.assert_array_equal(
            dst.states['s'], src.states['s'])
        self.assertIsNot(dst.states['s'], src.states['s'])

    def test_visitparams(self):
        model = self.model
        model.name = '/foo'
        model.grads['w'] = numpy.empty_like(model.params['w'])
        params = tuple(model.visitparams())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0][0], '/foo/_params/w')
        self.assertIs(params[0][1], model.params['w'])
        self.assertIs(params[0][2], model.grads['w'])

    def test_visitmodels(self):
        model = self.model
        models = tuple(model.visitmodels())
        self.assertEqual(len(models), 1)
        self.assertIs(models[0], model)

    def test_copyparams(self):
        src = Model1()
        dst = self.model
        w = numpy.ndarray(3, dtype='f')
        src.params['w'] = w
        w_dst = dst.params['w']

        dst.copyparams(src)
        numpy.testing.assert_array_equal(w_dst, w)
        self.assertIs(dst.params['w'], w_dst)

    def test_addgrads(self):
        g_src = numpy.array([1, 2, 3], dtype='f')
        g_dst = numpy.array([2, 3, 4], dtype='f')

        src = Model1()
        src.grads['w'] = g_src.copy()
        self.model.grads['w'] = g_dst.copy()
        self.model.addgrads(src)
        numpy.testing.assert_array_equal(
            self.model.grads['w'], g_src + g_dst)

    def test_zerograds(self):
        model = self.model
        self.assertNotIn('w', model.grads)
        model.zerograds()
        numpy.testing.assert_array_equal(
            model.grads['w'],
            numpy.zeros_like(model.params['w']))


class TestModelDict(unittest.TestCase):

    def setUp(self):
        self.model = chainer.ModelDict(
            ch1=Model1(),
            ch2=Model1(),
        )

    def test_elem_name(self):
        self.assertEqual(self.model['ch1'].name, '/ch1')
        self.assertEqual(self.model['ch2'].name, '/ch2')

    def test_init_non_model(self):
        self.assertRaises(TypeError, chainer.ModelDict, m1=0)

    def test_init_multiple_parents(self):
        m = Model1()
        m.name = '/foo'
        self.assertRaises(ValueError, chainer.ModelDict, bar=m)
        self.assertRaises(ValueError, chainer.ModelDict,
                          bar=self.model['ch1'])

    def test_contains(self):
        self.assertIn('ch1', self.model)
        self.assertIn('ch2', self.model)
        self.assertNotIn('ch3', self.model)
        self.assertNotIn('_params', self.model)

    def test_delitem(self):
        model = self.model['ch1']
        del self.model['ch1']
        self.assertNotIn('ch1', self.model)
        self.assertEqual(model.name, '/')

    def test_iter(self):
        d = dict(self.model)
        self.assertIn('ch1', d)
        self.assertIs(d['ch1'], self.model['ch1'])
        self.assertIn('ch2', d)
        self.assertIs(d['ch2'], self.model['ch2'])

    def test_setitem(self):
        old_model = self.model['ch1']
        new_model = Model1()
        self.model['ch1'] = new_model
        self.assertIs(self.model['ch1'], new_model)
        self.assertEqual(new_model.name, '/ch1')
        self.assertEqual(old_model.name, '/')

    def test_setitem_non_model(self):
        self.assertRaises(TypeError, self.model.__setitem__, 0)

    def test_setitem_multiple_parents(self):
        m = Model1()
        m.name = '/foo'
        self.assertRaises(ValueError, self.model.__setitem__, 'bar', m)

    def test_clear(self):
        ch1 = self.model['ch1']
        ch2 = self.model['ch2']
        self.model.clear()
        self.assertNotIn('ch1', self.model)
        self.assertNotIn('ch2', self.model)
        self.assertEqual(ch1.name, '/')
        self.assertEqual(ch2.name, '/')

    def test_get(self):
        self.assertIs(self.model.get('ch1'), self.model['ch1'])
        self.assertIs(self.model.get('ch3', None), None)

    def test_items(self):
        items = dict(self.model.items())
        self.assertIn('ch1', items)
        self.assertIs(items['ch1'], self.model['ch1'])
        self.assertIn('ch2', items)
        self.assertIs(items['ch2'], self.model['ch2'])

    if sys.version_info.major < 3:
        def test_iteritems(self):
            items = dict(self.model.iteritems())
            self.assertIn('ch1', items)
            self.assertIs(items['ch1'], self.model['ch1'])
            self.assertIn('ch2', items)
            self.assertIs(items['ch2'], self.model['ch2'])

        def test_iterkeys(self):
            keys = list(self.model.iterkeys())
            self.assertEqual(len(keys), 2)
            self.assertIn('ch1', keys)
            self.assertIn('ch2', keys)

        def test_itervalues(self):
            values = list(self.model.itervalues())
            self.assertEqual(len(values), 2)
            self.assertIn(self.model['ch1'], values)
            self.assertIn(self.model['ch2'], values)

    def test_has_key(self):
        self.assertTrue(self.model.has_key('ch1'))
        self.assertTrue(self.model.has_key('ch2'))
        self.assertFalse(self.model.has_key('ch3'))

    def test_keys(self):
        keys = list(self.model.keys())
        self.assertEqual(len(keys), 2)
        self.assertIn('ch1', keys)
        self.assertIn('ch2', keys)

    def test_pop(self):
        ch1 = self.model.pop('ch1')
        self.assertEqual(len(self.model), 1)
        self.assertNotIn('ch1', self.model)
        self.assertIn('ch2', self.model)
        self.assertEqual(ch1.name, '/')

        self.assertRaises(KeyError, self.model.pop, 'ch1')
        self.assertEqual(self.model.pop('ch1', 100), 100)

    def test_popitem(self):
        d = dict(self.model)
        k, m = self.model.popitem()
        self.assertIn(k, ('ch1', 'ch2'))
        self.assertIs(m, d[k])
        self.assertEqual(m.name, '/')

    def test_setdefault(self):
        ch1 = self.model['ch1']
        self.assertIs(self.model.setdefault('ch1'), ch1)
        self.assertIs(self.model['ch1'], ch1)

        m = Model1()
        ret = self.model.setdefault('ch3', m)
        self.assertIn('ch3', self.model)
        self.assertIs(self.model['ch3'], m)
        self.assertEqual(m.name, '/ch3')

        self.assertRaises(TypeError, self.model.setdefault, 'ch4')

        m2 = Model1()
        m2.name = '/foo'
        self.assertRaises(ValueError, self.model.setdefault, 'ch4', m2)

    def test_values(self):
        values = list(self.model.values())
        self.assertEqual(len(values), 2)
        self.assertIn(self.model['ch1'], values)
        self.assertIn(self.model['ch2'], values)

    def test_name(self):
        self.model.name = '/foo'
        self.assertEqual(self.model.name, '/foo')
        self.assertEqual(self.model['ch1'].name, '/foo/ch1')
        self.assertEqual(self.model['ch2'].name, '/foo/ch2')

    def test_visitmodels(self):
        models = list(self.model.visitmodels())
        self.assertEqual(len(models), 3)
        self.assertIs(models[0], self.model)
        self.assertIn(models[1], self.model.values())
        self.assertIn(models[2], self.model.values())


class TestModelList(unittest.TestCase):

    def setUp(self):
        self.model = chainer.ModelList(Model1(), Model1())

    def test_elem_name(self):
        self.assertEqual(self.model[0].name, '/0')
        self.assertEqual(self.model[1].name, '/1')

    def test_init_non_model(self):
        self.assertRaises(TypeError, chainer.ModelList, 0)

    def test_init_multiple_parents(self):
        m = Model1()
        m.name = '/foo'
        self.assertRaises(ValueError, chainer.ModelList, m)

    def test_getitem(self):
        m0 = Model1()
        m1 = Model1()
        l = chainer.ModelList(m0, m1)
        self.assertIs(l[0], m0)
        self.assertIs(l[1], m1)
        self.assertRaises(IndexError, l.__getitem__, 2)

    def test_iter(self):
        l = list(self.model)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], self.model[0])
        self.assertIs(l[1], self.model[1])

    def test_len(self):
        self.assertEqual(len(self.model), 2)

    def test_setitem(self):
        old = self.model[0]
        new = Model1()
        self.model[0] = new
        self.assertIs(self.model[0], new)
        self.assertEqual(new.name, '/0')
        self.assertEqual(old.name, '/')

    def test_setitem_non_model(self):
        self.assertRaises(TypeError, self.model.__setitem__, 0, 0)

    def test_setitem_multiple_parents(self):
        m = Model1()
        m.name = '/foo'
        self.assertRaises(ValueError, self.model.__setitem__, 0, m)

    def test_setitem_out_of_index(self):
        self.assertRaises(IndexError, self.model.__setitem__, 2, Model1())

    def test_append(self):
        m = Model1()
        self.model.append(m)
        self.assertEqual(len(self.model), 3)
        self.assertIs(self.model[2], m)
        self.assertEqual(m.name, '/2')

    def test_append_non_model(self):
        self.assertRaises(TypeError, self.model.append, 0)

    def test_append_multiple_parents(self):
        m = Model1()
        m.name = '/foo'
        self.assertRaises(ValueError, self.model.append, m)

    def test_pop(self):
        m0 = self.model[0]
        m1 = self.model[1]
        p = self.model.pop()
        self.assertIs(m1, p)
        self.assertEqual(len(self.model), 1)
        self.assertIs(self.model[0], m0)
        self.assertEqual(p.name, '/')

    def test_name(self):
        self.model.name = '/foo'
        self.assertEqual(self.model[0].name, '/foo/0')
        self.assertEqual(self.model[1].name, '/foo/1')

    def test_visitmodels(self):
        models = list(self.model.visitmodels())
        self.assertEqual(len(models), 3)
        self.assertIs(models[0], self.model)
        self.assertIs(models[1], self.model[0])
        self.assertIs(models[2], self.model[1])
