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


@pytest.mark.parametrize('datasets', [{'a': ds, 'b': 5}, {1: ds}, {'test': ds.Tair}])
def test_validation_error(datasets):
    with pytest.raises(pydantic.ValidationError):
        xcollection.Collection(datasets)


@pytest.mark.parametrize('value', [1, ds.Tair, 'test'])
def test_setitem_validation(value):
    c = xcollection.Collection()
    with pytest.raises(TypeError):
        c['my_key'] = value
