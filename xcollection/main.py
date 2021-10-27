import typing
from collections.abc import MutableMapping
from typing import Hashable, Iterable, Optional, Union

import pydantic
import toolz
import xarray as xr
from xarray.core.weighted import Weighted


def _rpartial(func, *args, **kwargs):
    """Partially applies last arguments.
    New keyworded arguments extend and override kwargs.

    Notes
    -----
    This code is copied from https://github.com/Suor/funcy,
    which is available under the BSD 3-Clause License.
    """
    return lambda *a, **kw: func(*(a + args), **dict(kwargs, **kw))


def _validate_input(value):
    if not isinstance(
        value,
        (
            xr.Dataset,
            xr.DataArray,
            xr.core.weighted.DataArrayWeighted,
            xr.core.weighted.DatasetWeighted,
        ),
    ):
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

    def choose(
        self, data_vars: typing.Union[str, typing.List[str]], *, mode: str = 'any'
    ) -> 'Collection':
        """Return a collection with datasets containing all or any of the specified data variables.
        Parameters
        ----------
        data_vars : str or list of str
            The data variables to select on.
        mode : str, optional
            The selection mode. Must be one of 'all' or 'any'. Defaults to 'any'.

        Returns
        -------
        Collection
            A new collection containing only the selected datasets.

        """

        _VALID_MODES = ['all', 'any']
        if mode not in _VALID_MODES:
            raise ValueError(f'Invalid mode: {mode}. Accepted modes are {_VALID_MODES}')

        if isinstance(data_vars, str):
            data_vars = [data_vars]

        def _select_vars(dset):
            try:
                return dset[data_vars]
            except KeyError:
                if mode == 'all':
                    raise KeyError(f'No data variables: `{data_vars}` found in dataset: {dset!r}')

        if mode == 'all':
            result = toolz.valmap(_select_vars, self.datasets)
        elif mode == 'any':
            result = toolz.valfilter(_select_vars, self.datasets)

        return type(self)(datasets=result)

    def keymap(self, func: typing.Callable[[str], str]) -> 'Collection':
        """Apply a function to each key in the collection.
        Parameters
        ----------
        func : callable
            The function to apply to each key.

        Returns
        -------
        Collection
            A new collection containing the results of the function.

        """
        if not callable(func):
            raise TypeError(f'First argument must be callable function, got {type(func)}')

        return type(self)(datasets=toolz.keymap(func, self.datasets))

    def map(
        self,
        func: typing.Callable[[xr.Dataset], xr.Dataset],
        args: typing.Sequence[typing.Any] = (),
        **kwargs: typing.Dict[str, typing.Any],
    ) -> 'Collection':
        """Apply a function to each dataset in the collection.
        Parameters
        ----------
        func : callable
            The function to apply to each dataset.
        args : tuple, optional
            Positional arguments to pass to `func` in addition to the
            dataset.
        kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        Collection
            A new collection containing the results of the function.

        """
        if not callable(func):
            raise TypeError(f'First argument must be callable function, got {type(func)}')

        if not isinstance(args, tuple):
            raise TypeError(f'Second argument must be a tuple, got {type(args)}')

        func = _rpartial(func, *args, **kwargs)
        return type(self)(datasets=toolz.valmap(func, self.datasets))

    def weighted(self, weights, **kwargs) -> 'Collection':
        return CollectionWeighted(self, weights, *kwargs)


class CollectionWeighted(Weighted['Collection']):
    def _check_dim(self, dim: Optional[Union[Hashable, Iterable[Hashable]]]):
        """raise an error if any dimension is missing"""

        for key, dataset in self.obj.items():
            if isinstance(dim, str) or not isinstance(dim, Iterable):
                dims = [dim] if dim else []
            else:
                dims = list(dim)
            missing_dims = set(dims) - set(dataset.dims) - set(self.weights.dims)
            if missing_dims:
                raise ValueError(
                    f'{dataset.__class__.__name__} does not contain the dimensions: {missing_dims}'
                )

    def _implementation(self, func, dim, **kwargs) -> 'Collection':

        self._check_dim(dim)

        dataset_dict = {}
        for key, dataset in self.obj.items():

            dataset = dataset.map(func, dim=dim, **kwargs)
            dataset_dict[key] = dataset
        return Collection(dataset_dict)
