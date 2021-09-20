from typing import TYPE_CHECKING, TypeVar

from xarray.core import types

if TYPE_CHECKING:
    from xarray import DataArray, Dataset

    from .main import Collection


types.T_Xarray = TypeVar('T_Xarray', 'DataArray', 'Dataset', 'Collection')
T_Collection = TypeVar('T_Collection', bound='Collection')
