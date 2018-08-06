import pytest

import xchainer


def test_no_backprop_mode():
    with xchainer.backprop_scope('bp1') as bp1, \
            xchainer.backprop_scope('bp2') as bp2:
        assert xchainer.is_backprop_required()
        assert xchainer.is_backprop_required(bp1)
        assert xchainer.is_backprop_required(bp2)

        with xchainer.no_backprop_mode():
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)
        assert xchainer.is_backprop_required()
        assert xchainer.is_backprop_required(bp1)
        assert xchainer.is_backprop_required(bp2)

        with xchainer.no_backprop_mode(xchainer.get_default_context()):
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)
        assert xchainer.is_backprop_required()
        assert xchainer.is_backprop_required(bp1)
        assert xchainer.is_backprop_required(bp2)

        with xchainer.no_backprop_mode(bp1):
            assert xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert xchainer.is_backprop_required(bp2)
        assert xchainer.is_backprop_required()
        assert xchainer.is_backprop_required(bp1)
        assert xchainer.is_backprop_required(bp2)

        with xchainer.no_backprop_mode((bp1, bp2)):
            assert xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)
        assert xchainer.is_backprop_required()
        assert xchainer.is_backprop_required(bp1)
        assert xchainer.is_backprop_required(bp2)


def test_force_backprop_mode():
    with xchainer.backprop_scope('bp1') as bp1, \
            xchainer.backprop_scope('bp2') as bp2:
        with xchainer.no_backprop_mode():
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)

            with xchainer.force_backprop_mode():
                assert xchainer.is_backprop_required()
                assert xchainer.is_backprop_required(bp1)
                assert xchainer.is_backprop_required(bp2)
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)

            with xchainer.force_backprop_mode(xchainer.get_default_context()):
                assert xchainer.is_backprop_required()
                assert xchainer.is_backprop_required(bp1)
                assert xchainer.is_backprop_required(bp2)
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)

            with xchainer.force_backprop_mode(bp1):
                assert not xchainer.is_backprop_required()
                assert xchainer.is_backprop_required(bp1)
                assert not xchainer.is_backprop_required(bp2)
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)

            with xchainer.force_backprop_mode((bp1, bp2)):
                assert not xchainer.is_backprop_required()
                assert xchainer.is_backprop_required(bp1)
                assert xchainer.is_backprop_required(bp2)
            assert not xchainer.is_backprop_required()
            assert not xchainer.is_backprop_required(bp1)
            assert not xchainer.is_backprop_required(bp2)

        with xchainer.force_backprop_mode():
            assert xchainer.is_backprop_required()
            assert xchainer.is_backprop_required(bp1)
            assert xchainer.is_backprop_required(bp2)


def test_is_backprop_required():
    current_context = xchainer.get_default_context()
    another_context = xchainer.Context()

    with xchainer.backprop_scope('bp1') as bp1, \
            xchainer.backprop_scope('bp2') as bp2:
        with xchainer.no_backprop_mode():
            with xchainer.force_backprop_mode(bp1):
                assert not xchainer.is_backprop_required()
                assert xchainer.is_backprop_required(bp1)
                assert not xchainer.is_backprop_required(bp2)
                assert not xchainer.is_backprop_required(context=current_context)
                assert xchainer.is_backprop_required(context=another_context)

        with pytest.raises(TypeError):
            xchainer.is_backprop_required(context='foo')
