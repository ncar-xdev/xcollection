import json
import os
import pathlib

import xarray as xr

from .main import Collection


def read_collection(group_path, engine=None):
    collection_dict = dict()

    if os.path.exists(f'{group_path}/.xcollection.json'):
        metadata_file = open(f'{group_path}/.xcollection.json', 'r')
    else:
        raise ValueError(f'{group_path} missing metadata file')

    metadata_dict = json.loads(metadata_file.read())

    if metadata_dict['file_extension'] == '.nc':
        engine = 'netcdf4'

    elif metadata_dict['file_extension'] == '.zarr':
        engine = 'zarr'

    else:
        raise ('File format not supported')

    for file in metadata_dict['file_list']:
        ds = xr.open_dataset(file, engine=engine)
        stem = pathlib.Path(file).stem
        collection_dict[stem] = ds

    return Collection(collection_dict)
