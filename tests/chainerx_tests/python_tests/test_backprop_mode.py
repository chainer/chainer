import pytest

import chainerx


def test_no_backprop_mode():
    with chainerx.backprop_scope('bp1') as bp1, \
            chainerx.backprop_scope('bp2') as bp2:
        assert chainerx.is_backprop_required()
        assert chainerx.is_backprop_required(bp1)
        assert chainerx.is_backprop_required(bp2)

        with chainerx.no_backprop_mode():
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)
        assert chainerx.is_backprop_required()
        assert chainerx.is_backprop_required(bp1)
        assert chainerx.is_backprop_required(bp2)

        with chainerx.no_backprop_mode(chainerx.get_default_context()):
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)
        assert chainerx.is_backprop_required()
        assert chainerx.is_backprop_required(bp1)
        assert chainerx.is_backprop_required(bp2)

        with chainerx.no_backprop_mode(bp1):
            assert chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert chainerx.is_backprop_required(bp2)
        assert chainerx.is_backprop_required()
        assert chainerx.is_backprop_required(bp1)
        assert chainerx.is_backprop_required(bp2)

        with chainerx.no_backprop_mode((bp1, bp2)):
            assert chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)
        assert chainerx.is_backprop_required()
        assert chainerx.is_backprop_required(bp1)
        assert chainerx.is_backprop_required(bp2)


def test_force_backprop_mode():
    with chainerx.backprop_scope('bp1') as bp1, \
            chainerx.backprop_scope('bp2') as bp2:
        with chainerx.no_backprop_mode():
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)

            with chainerx.force_backprop_mode():
                assert chainerx.is_backprop_required()
                assert chainerx.is_backprop_required(bp1)
                assert chainerx.is_backprop_required(bp2)
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)

            with chainerx.force_backprop_mode(chainerx.get_default_context()):
                assert chainerx.is_backprop_required()
                assert chainerx.is_backprop_required(bp1)
                assert chainerx.is_backprop_required(bp2)
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)

            with chainerx.force_backprop_mode(bp1):
                assert not chainerx.is_backprop_required()
                assert chainerx.is_backprop_required(bp1)
                assert not chainerx.is_backprop_required(bp2)
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)

            with chainerx.force_backprop_mode((bp1, bp2)):
                assert not chainerx.is_backprop_required()
                assert chainerx.is_backprop_required(bp1)
                assert chainerx.is_backprop_required(bp2)
            assert not chainerx.is_backprop_required()
            assert not chainerx.is_backprop_required(bp1)
            assert not chainerx.is_backprop_required(bp2)

        with chainerx.force_backprop_mode():
            assert chainerx.is_backprop_required()
            assert chainerx.is_backprop_required(bp1)
            assert chainerx.is_backprop_required(bp2)


def test_is_backprop_required():
    current_context = chainerx.get_default_context()
    another_context = chainerx.Context()

    with chainerx.backprop_scope('bp1') as bp1, \
            chainerx.backprop_scope('bp2') as bp2:
        with chainerx.no_backprop_mode():
            with chainerx.force_backprop_mode(bp1):
                assert not chainerx.is_backprop_required()
                assert chainerx.is_backprop_required(bp1)
                assert not chainerx.is_backprop_required(bp2)
                assert not chainerx.is_backprop_required(context=current_context)
                assert chainerx.is_backprop_required(context=another_context)

        with pytest.raises(TypeError):
            chainerx.is_backprop_required(context='foo')
