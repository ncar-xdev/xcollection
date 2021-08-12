import typing
from collections.abc import MutableMapping

import pydantic
import xarray as xr


def _validate_input(value):
    if not isinstance(value, (xr.Dataset, xr.DataArray)):
        raise TypeError(f'Expected an xarray.Dataset or xarray.DataArray, got {type(value)}')
    if isinstance(value, xr.DataArray):
        return value.to_dataset()
    return value


class Config:
    validate_assignment = True
    arbitrary_types_allowed = True


@pydantic.dataclasses.dataclass(config=Config)
class Collection(MutableMapping):
    datasets: typing.Dict[pydantic.StrictStr, typing.Union[xr.Dataset, xr.DataArray]] = None

    @pydantic.validator('datasets', pre=True, each_item=True)
    def _validate_datasets(cls, value):
        return _validate_input(value)

    def __post_init_post_parse__(self):
        if self.datasets is None:
            self.datasets = {}

    def __delitem__(self, key: str) -> None:
        del self.datasets[key]

    def __getitem__(self, key: str) -> xr.Dataset:
        try:
            return self.datasets[key]
        except KeyError:
            raise KeyError(f'Dataset with key: `{key}` not found')

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    def __setitem__(self, key: str, value: xr.Dataset) -> None:
        self.datasets[key] = _validate_input(value)

    def __contains__(self, key: str) -> bool:
        return key in self.datasets

    def __repr__(self) -> str:
        unicode_key = u'\U0001F511'
        output = ''.join(f'{unicode_key} {key}\n{repr(value)}\n\n' for key, value in self.items())
        return f'<{type(self).__name__} ({len(self)} keys)>\n{output}'

    def _repr_html_(self):
        """
        Return an html representation for the collection object.
        Mainly for IPython notebook
        """
        # TODO: Extend Xarray HTML wrapper to output collapsible dataset entries

    def _ipython_display_(self):
        """
        Display the collection in the IPython notebook.
        """
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))

    def keys(self) -> typing.Iterable[str]:
        return self.datasets.keys()

    def values(self) -> typing.Iterable[xr.Dataset]:
        return self.datasets.values()

    def items(self) -> typing.Iterable[typing.Tuple[str, xr.Dataset]]:
        return self.datasets.items()
