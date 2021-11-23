import glob
import pathlib

import xarray as xr

from .main import Collection


def read_collection(group_path, engine=None):
    collection_dict = dict()

    if engine == 'netcdf4':
        extension = 'nc'

    elif engine == 'zarr':
        extension = 'zarr'

    else:
        raise ('File format not supported')

    files = glob.glob(f'{group_path}/*{extension}')

    for file in files:
        ds = xr.open_dataset(file, engine=engine)
        stem = pathlib.Path(file).stem
        collection_dict[stem] = ds

    return Collection(collection_dict)
