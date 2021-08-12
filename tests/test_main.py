import typing

import pydantic
import pytest
import xarray as xr

import xcollection

ds = xr.tutorial.open_dataset('rasm')


@pytest.mark.parametrize('datasets', [None, {'a': ds, 'b': ds}])
def test_init(datasets):
    c = xcollection.Collection(datasets)
    assert isinstance(c.datasets, dict)
    if datasets is not None:
        assert len(c) == len(datasets)
        assert set(c.keys()) == set(datasets.keys())


def test_repr():
    c = xcollection.Collection({'bar': ds, 'foo': ds})
    assert 'foo' in repr(c)
    assert 'bar' in repr(c)


@pytest.mark.parametrize('datasets', [{'a': ds, 'b': 5}, {1: ds}, {'test': ds.Tair}])
def test_validation_error(datasets):
    with pytest.raises(pydantic.ValidationError):
        xcollection.Collection(datasets)


@pytest.mark.parametrize('value', [1, ds.Tair, 'test'])
def test_setitem_validation(value):
    c = xcollection.Collection()
    with pytest.raises(TypeError):
        c['my_key'] = value


def test_setitem():
    c = xcollection.Collection()
    c['my_key'] = ds
    assert len(c) == 1
    assert 'my_key' in c


def test_delitem():
    c = xcollection.Collection({'a': ds, 'b': ds})
    assert len(c) == 2
    assert set(c.keys()) == {'a', 'b'}
    del c['a']
    assert len(c) == 1
    assert set(c.keys()) == {'b'}


def test_getitem():
    datasets = {'a': ds, 'b': ds}
    c = xcollection.Collection(datasets)
    assert isinstance(c['a'], xr.Dataset)

    with pytest.raises(KeyError):
        c['foo']


def test_iter():
    c = xcollection.Collection()
    assert isinstance(iter(c), typing.Iterator)
