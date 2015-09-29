import sys
import unittest

import numpy

import chainer


class Link1(chainer.Link):

    def __init__(self):
        super(Link1, self).__init__()
        self.params['w'] = chainer.Variable(numpy.array([1, 2, 3], dtype='f'))
        self.states['s'] = numpy.array([4, 5, 6], dtype='f')


class TestLink(unittest.TestCase):

    def setUp(self):
        self.link = Link1()

    def test_name(self):
        self.assertEqual(self.link.name, '/')
        self.link.name = '/foo'
        self.assertEqual(self.link.name, '/foo')

    def test_copy_shared(self):
        src = self.link
        src.name = '/foo'
        dst = src.copy()

        self.assertEqual(dst.name, src.name)
        self.assertIsNot(dst.params, src.params)
        self.assertIsNot(dst.states, src.states)
        self.assertIsNot(dst.params['w'], src.params['w'])
        self.assertIs(dst.params['w'].data, src.params['w'].data)
        self.assertIs(dst.states['s'], src.states['s'])

    def test_copy_not_shared(self):
        src = self.link
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
        link = self.link
        link.name = '/foo'
        w = link.params['w']
        w.grad = numpy.empty_like(w.data)
        params = tuple(link.visitparams())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0][0], '/foo/_params/w')
        self.assertIs(params[0][1], w)

    def test_visitlinks(self):
        link = self.link
        links = tuple(link.visitlinks())
        self.assertEqual(len(links), 1)
        self.assertIs(links[0], link)

    def test_copyparams(self):
        src = Link1()
        dst = self.link
        w = chainer.Variable(numpy.ndarray(3, dtype='f'))
        src.params['w'] = w
        w_dst = dst.params['w']

        dst.copyparams(src)
        numpy.testing.assert_array_equal(w_dst.data, w.data)
        self.assertIs(dst.params['w'].data, w_dst.data)

    def test_addgrads(self):
        g_src = numpy.array([1, 2, 3], dtype='f')
        g_dst = numpy.array([2, 3, 4], dtype='f')

        src = Link1()
        src.params['w'].grad = g_src.copy()
        self.link.params['w'].grad = g_dst.copy()
        self.link.addgrads(src)
        numpy.testing.assert_array_equal(
            self.link.params['w'].grad, g_src + g_dst)

    def test_zerograds(self):
        link = self.link
        self.assertIs(link.params['w'].grad, None)
        link.zerograds()
        numpy.testing.assert_array_equal(
            link.params['w'].grad,
            numpy.zeros_like(link.params['w'].data))


class TestDictLink(unittest.TestCase):

    def setUp(self):
        self.link = chainer.DictLink(
            ch1=Link1(),
            ch2=Link1(),
        )

    def test_elem_name(self):
        self.assertEqual(self.link['ch1'].name, '/ch1')
        self.assertEqual(self.link['ch2'].name, '/ch2')

    def test_init_non_model(self):
        self.assertRaises(TypeError, chainer.DictLink, o1=0)

    def test_init_multiple_parents(self):
        o = Link1()
        o.name = '/foo'
        self.assertRaises(ValueError, chainer.DictLink, bar=o)
        self.assertRaises(ValueError, chainer.DictLink,
                          bar=self.link['ch1'])

    def test_contains(self):
        self.assertIn('ch1', self.link)
        self.assertIn('ch2', self.link)
        self.assertNotIn('ch3', self.link)
        self.assertNotIn('_params', self.link)

    def test_delitem(self):
        model = self.link['ch1']
        del self.link['ch1']
        self.assertNotIn('ch1', self.link)
        self.assertEqual(model.name, '/')

    def test_iter(self):
        d = dict(self.link)
        self.assertIn('ch1', d)
        self.assertIs(d['ch1'], self.link['ch1'])
        self.assertIn('ch2', d)
        self.assertIs(d['ch2'], self.link['ch2'])

    def test_setitem(self):
        old_link = self.link['ch1']
        new_link = Link1()
        self.link['ch1'] = new_link
        self.assertIs(self.link['ch1'], new_link)
        self.assertEqual(new_link.name, '/ch1')
        self.assertEqual(old_link.name, '/')

    def test_setitem_non_model(self):
        self.assertRaises(TypeError, self.link.__setitem__, 0)

    def test_setitem_multiple_parents(self):
        o = Link1()
        o.name = '/foo'
        self.assertRaises(ValueError, self.link.__setitem__, 'bar', o)

    def test_clear(self):
        ch1 = self.link['ch1']
        ch2 = self.link['ch2']
        self.link.clear()
        self.assertNotIn('ch1', self.link)
        self.assertNotIn('ch2', self.link)
        self.assertEqual(ch1.name, '/')
        self.assertEqual(ch2.name, '/')

    def test_get(self):
        self.assertIs(self.link.get('ch1'), self.link['ch1'])
        self.assertIs(self.link.get('ch3', None), None)

    def test_items(self):
        items = dict(self.link.items())
        self.assertIn('ch1', items)
        self.assertIs(items['ch1'], self.link['ch1'])
        self.assertIn('ch2', items)
        self.assertIs(items['ch2'], self.link['ch2'])

    if sys.version_info.major < 3:
        def test_iteritems(self):
            items = dict(self.link.iteritems())
            self.assertIn('ch1', items)
            self.assertIs(items['ch1'], self.link['ch1'])
            self.assertIn('ch2', items)
            self.assertIs(items['ch2'], self.link['ch2'])

        def test_iterkeys(self):
            keys = list(self.link.iterkeys())
            self.assertEqual(len(keys), 2)
            self.assertIn('ch1', keys)
            self.assertIn('ch2', keys)

        def test_itervalues(self):
            values = list(self.link.itervalues())
            self.assertEqual(len(values), 2)
            self.assertIn(self.link['ch1'], values)
            self.assertIn(self.link['ch2'], values)

    def test_has_key(self):
        self.assertTrue(self.link.has_key('ch1'))
        self.assertTrue(self.link.has_key('ch2'))
        self.assertFalse(self.link.has_key('ch3'))

    def test_keys(self):
        keys = list(self.link.keys())
        self.assertEqual(len(keys), 2)
        self.assertIn('ch1', keys)
        self.assertIn('ch2', keys)

    def test_pop(self):
        ch1 = self.link.pop('ch1')
        self.assertEqual(len(self.link), 1)
        self.assertNotIn('ch1', self.link)
        self.assertIn('ch2', self.link)
        self.assertEqual(ch1.name, '/')

        self.assertRaises(KeyError, self.link.pop, 'ch1')
        self.assertEqual(self.link.pop('ch1', 100), 100)

    def test_popitem(self):
        d = dict(self.link)
        k, o = self.link.popitem()
        self.assertIn(k, ('ch1', 'ch2'))
        self.assertIs(o, d[k])
        self.assertEqual(o.name, '/')

    def test_setdefault(self):
        ch1 = self.link['ch1']
        self.assertIs(self.link.setdefault('ch1'), ch1)
        self.assertIs(self.link['ch1'], ch1)

        o = Link1()
        ret = self.link.setdefault('ch3', o)
        self.assertIn('ch3', self.link)
        self.assertIs(self.link['ch3'], o)
        self.assertEqual(o.name, '/ch3')

        self.assertRaises(TypeError, self.link.setdefault, 'ch4')

        o2 = Link1()
        o2.name = '/foo'
        self.assertRaises(ValueError, self.link.setdefault, 'ch4', o2)

    def test_values(self):
        values = list(self.link.values())
        self.assertEqual(len(values), 2)
        self.assertIn(self.link['ch1'], values)
        self.assertIn(self.link['ch2'], values)

    def test_name(self):
        self.link.name = '/foo'
        self.assertEqual(self.link.name, '/foo')
        self.assertEqual(self.link['ch1'].name, '/foo/ch1')
        self.assertEqual(self.link['ch2'].name, '/foo/ch2')

    def test_visitlinks(self):
        links = list(self.link.visitlinks())
        self.assertEqual(len(links), 3)
        self.assertIs(links[0], self.link)
        self.assertIn(links[1], self.link.values())
        self.assertIn(links[2], self.link.values())


class TestListLink(unittest.TestCase):

    def setUp(self):
        self.link = chainer.ListLink(Link1(), Link1())

    def test_elem_name(self):
        self.assertEqual(self.link[0].name, '/0')
        self.assertEqual(self.link[1].name, '/1')

    def test_init_non_model(self):
        self.assertRaises(TypeError, chainer.ListLink, 0)

    def test_init_multiple_parents(self):
        o = Link1()
        o.name = '/foo'
        self.assertRaises(ValueError, chainer.ListLink, o)

    def test_getitem(self):
        o0 = Link1()
        o1 = Link1()
        l = chainer.ListLink(o0, o1)
        self.assertIs(l[0], o0)
        self.assertIs(l[1], o1)
        self.assertRaises(IndexError, l.__getitem__, 2)

    def test_iter(self):
        l = list(self.link)
        self.assertEqual(len(l), 2)
        self.assertIs(l[0], self.link[0])
        self.assertIs(l[1], self.link[1])

    def test_len(self):
        self.assertEqual(len(self.link), 2)

    def test_setitem(self):
        old = self.link[0]
        new = Link1()
        self.link[0] = new
        self.assertIs(self.link[0], new)
        self.assertEqual(new.name, '/0')
        self.assertEqual(old.name, '/')

    def test_setitem_non_model(self):
        self.assertRaises(TypeError, self.link.__setitem__, 0, 0)

    def test_setitem_multiple_parents(self):
        o = Link1()
        o.name = '/foo'
        self.assertRaises(ValueError, self.link.__setitem__, 0, o)

    def test_setitem_out_of_index(self):
        self.assertRaises(IndexError, self.link.__setitem__, 2, Link1())

    def test_append(self):
        o = Link1()
        self.link.append(o)
        self.assertEqual(len(self.link), 3)
        self.assertIs(self.link[2], o)
        self.assertEqual(o.name, '/2')

    def test_append_non_model(self):
        self.assertRaises(TypeError, self.link.append, 0)

    def test_append_multiple_parents(self):
        o = Link1()
        o.name = '/foo'
        self.assertRaises(ValueError, self.link.append, o)

    def test_pop(self):
        o0 = self.link[0]
        o1 = self.link[1]
        p = self.link.pop()
        self.assertIs(o1, p)
        self.assertEqual(len(self.link), 1)
        self.assertIs(self.link[0], o0)
        self.assertEqual(p.name, '/')

    def test_name(self):
        self.link.name = '/foo'
        self.assertEqual(self.link[0].name, '/foo/0')
        self.assertEqual(self.link[1].name, '/foo/1')

    def test_visitlinks(self):
        links = list(self.link.visitlinks())
        self.assertEqual(len(links), 3)
        self.assertIs(links[0], self.link)
        self.assertIs(links[1], self.link[0])
        self.assertIs(links[2], self.link[1])
