import typing

import pydantic
import pytest
import xarray as xr
import zarr

import xcollection

ds = xr.tutorial.open_dataset('rasm')
dsa = xr.tutorial.open_dataset('air_temperature')


@pytest.mark.parametrize('datasets', [None, {'a': ds, 'b': ds}, {'test': ds.Tair}])
def test_init(datasets):
    c = xcollection.Collection(datasets)
    assert isinstance(c.datasets, dict)
    if datasets is not None:
        assert len(c) == len(datasets)
        assert set(c.keys()) == set(datasets.keys())
    else:
        assert len(c) == 0


def test_repr():
    c = xcollection.Collection({'bar': ds, 'foo': ds})
    assert 'foo' in repr(c)
    assert 'bar' in repr(c)


def test_html_repr():
    c = xcollection.Collection({'foo': ds, 'bar': ds})
    formatted = c._repr_html_()
    assert formatted.count("<li class='xr-var-item'><strong>") == len(c)


@pytest.mark.parametrize('datasets', [{'a': ds, 'b': 5}, {1: ds}])
def test_validation_error(datasets):
    with pytest.raises(pydantic.ValidationError):
        xcollection.Collection(datasets)


@pytest.mark.parametrize('value', [1, ds.coords, 'test'])
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
    assert c['a'] is ds
    assert c['b'] is ds

    with pytest.raises(KeyError):
        c['foo']


def test_iter():
    c = xcollection.Collection()
    assert isinstance(iter(c), typing.Iterator)


@pytest.mark.parametrize('data_vars', ['Tair', ['Tair']])
def test_choose_all(data_vars):
    c = xcollection.Collection({'foo': ds, 'bar': ds})
    d = c.choose(data_vars, mode='all')
    assert c == d
    assert set(d.keys()) == {'foo', 'bar'}


def test_choose_all_error():
    c = xcollection.Collection({'foo': ds, 'bar': dsa})
    with pytest.raises(KeyError):
        c.choose('Tair', mode='all')


def test_choose_mode_error():
    c = xcollection.Collection()
    with pytest.raises(ValueError):
        c.choose('Tair', mode='foo')


@pytest.mark.parametrize('data_vars', ['Tair', ['air']])
def test_choose_any(data_vars):
    c = xcollection.Collection({'foo': ds, 'bar': dsa})
    d = c.choose(data_vars, mode='any')
    assert len(d) == 1


@pytest.mark.parametrize('dim, attrs', [('time', {'foo': 'bar'}), (['lat', 'lon'], {})])
def test_map(dim, attrs):
    c = xcollection.Collection({'foo': dsa, 'bar': dsa})

    def func(ds, variable, attrs=None, dim=None):
        result = ds[variable].mean(dim)
        result.attrs = attrs
        return result

    d = c.map(func, args=('air',), attrs=attrs, dim=dim)
    assert set(c.keys()) == set(d.keys())
    xr.testing.assert_identical(d['foo'], func(dsa, 'air', attrs=attrs, dim=dim).to_dataset())


def test_keymap():
    c = xcollection.Collection({'foo': ds, 'bar': ds})
    d = c.keymap(lambda k: k.upper())
    assert set(d.keys()) == {'FOO', 'BAR'}

    with pytest.raises(TypeError):
        c.keymap('TEST')


def test_map_type_error():
    c = xcollection.Collection()
    with pytest.raises(TypeError):
        c.map('func')

    with pytest.raises(TypeError):
        c.map(lambda x: x, args=('foo'))


@pytest.mark.parametrize(
    'by, func, expected',
    [
        ('key', lambda x: isinstance(x, str), 2),
        ('value', lambda ds: 'Tair' in ds.data_vars, 1),
        ('item', lambda item: 2014 in item[1].time.dt.year and isinstance(item[0], str), 1),
    ],
)
def test_filter(by, func, expected):
    c = xcollection.Collection({'foo': ds, 'bar': dsa})
    d = c.filter(by=by, func=func)
    assert len(d) == expected


def test_filter_error():
    c = xcollection.Collection()
    with pytest.raises(ValueError):
        c.filter(by='foo', func=lambda x: x)


def test_equivalence():
    a = xcollection.Collection({'foo': ds, 'bar': ds})
    b = a.filter(by='key', func=lambda x: x == 'foo')
    c = xcollection.Collection({'foo': ds, 'bar': ds.isel(time=0)})
    assert a != ds
    assert a == a
    assert a != b
    assert a != c


def test_to_zarr(tmp_path):
    c = xcollection.Collection({'foo': ds.isel(time=0), 'bar': ds.isel(y=0)})
    store = str(tmp_path / 'testing.zarr')
    c.to_zarr(store)

    zstore = zarr.open_group(store, mode='r')
    assert set(zstore.group_keys()) == set(c.keys())

    c2 = xcollection.open_collection(store)
    assert c == c2


@pytest.mark.parametrize('datasets', [{'foo': ds, 'bar': dsa}])
def test_weighted(datasets):
    ds_dict = datasets
    collection = xcollection.Collection(ds_dict)

    a = list(ds_dict.keys())[0]
    b = list(ds_dict[a].data_vars)[0]
    weights = ds_dict[a][b].fillna(0)
    weights.name = 'weights'

    collection_weighted = collection.weighted(weights)
    collection_dict = {
        'mean': collection_weighted.mean(dim='time'),
        'sum': collection_weighted.sum(dim='time'),
        'sum_of_weights': collection_weighted.sum_of_weights(dim='time'),
    }

    dict_weighted = {key: ds_dict[key].weighted(weights) for key in ds_dict}
    dict_dict = {
        'mean': {key: dict_weighted[key].mean(dim='time') for key in dict_weighted},
        'sum': {key: dict_weighted[key].sum(dim='time') for key in dict_weighted},
        'sum_of_weights': {
            key: dict_weighted[key].sum_of_weights(dim='time') for key in dict_weighted
        },
    }

    for k in collection_dict:
        for j, ds in collection_dict[k].items():
            assert isinstance(ds, xr.Dataset)
            assert isinstance(dict_dict[k][j], xr.Dataset)
            assert ds == dict_dict[k][j]
