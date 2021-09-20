from xarray.core import types
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import Dataset, DataArray
    from .main import Collection
    
    
types.T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset", "Collection")
T_Collection = TypeVar("T_Collection", bound="Collection")