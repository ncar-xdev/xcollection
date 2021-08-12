import typing
from collections.abc import MutableMapping

import pydantic
import xarray as xr


class Config:
    validate_assignment = True
    arbitrary_types_allowed = True


@pydantic.dataclasses.dataclass(config=Config)
class Collection(MutableMapping):
    datasets: typing.Dict[pydantic.StrictStr, xr.Dataset] = None

    @pydantic.validator('datasets', pre=True, each_item=True)
    def _validate_dataset(cls, value):
        if not isinstance(value, xr.Dataset):
            raise TypeError(f'Expected an xarray.Dataset, got {type(value)}')
        return value

    def __post_init_post_parse__(self):
        if self.datasets is None:
            self.datasets = {}

    def __delitem__(self, key: str):
        del self.datasets[key]

    def __getitem__(self, key: str):
        return self.datasets[key]

    def __iter__(self):
        return iter(self.datasets)

    def __len__(self):
        return len(self.datasets)

    def __setitem__(self, key: str, value: xr.Dataset):
        if not isinstance(value, xr.Dataset):
            raise TypeError(f'Expected an xarray.Dataset, got {type(value)}')
        self.datasets[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.datasets

    def __repr__(self):
        unicode_key = u'\U0001F511'
        output = ''.join(f'{unicode_key} {key}\n{repr(value)}\n\n' for key, value in self.items())
        return f'<{type(self).__name__} ({len(self)} keys)>\n{output}'

    def _repr_html_(self):
        """
        Return an html representation for the collection object.
        Mainly for IPython notebook
        """

    def _ipython_display_(self):
        """
        Display the collection in the IPython notebook.
        """
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))

    def keys(self):
        return self.datasets.keys()

    def values(self):
        return self.datasets.values()

    def items(self):
        return self.datasets.items()
