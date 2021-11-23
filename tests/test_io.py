import os

import xarray as xr

import xcollection

ds = xr.tutorial.open_dataset('rasm')


def test_read_zarr():
    c = xcollection.Collection({'foo': ds, 'bar': ds})
    current_dir = os.getcwd()
    group_path = f'{current_dir}/test_dir'
    if not os.path.isdir(group_path):
        os.makedirs(group_path)
    c.save(group_path, format='zarr')
    new_c = xcollection.read_collection(group_path, engine='zarr')
    assert isinstance(new_c, xcollection.Collection)
    assert 'foo' in new_c.keys()


def test_read_netcdf():
    c = xcollection.Collection({'foo': ds, 'bar': ds})
    current_dir = os.getcwd()
    group_path = f'{current_dir}/test_dir'
    if not os.path.isdir(group_path):
        os.makedirs(group_path)
    c.save(group_path, format='nc')
    new_c = xcollection.read_collection(group_path, engine='netcdf4')
    assert isinstance(new_c, xcollection.Collection)
    assert 'foo' in new_c.keys()
