import functools
import typing
from collections.abc import MutableMapping
from html import escape
from typing import Hashable, Iterable, Optional, Union

import pydantic
import toolz
import xarray as xr
from xarray.core.weighted import Weighted

unicode_key = u'\U0001F511'


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
    """A collection of datasets. The keys are the dataset names and the values are the datasets.

    Parameters
    ----------
    datasets : dict, optional
        A dictionary of datasets to initialize the collection with.

    Examples
    --------
    >>> import xcollection as xc
    >>> import xarray as xr
    >>> ds = xr.tutorial.open_dataset('rasm')
    >>> c = xc.Collection({'foo': ds.isel(time=0), 'bar': ds.isel(y=0)})
    >>> c
    <Collection (2 keys)>
    🔑 foo
    <xarray.Dataset>
    Dimensions:  (y: 205, x: 275)
    Coordinates:
        time     object 1980-09-16 12:00:00
        xc       (y, x) float64 ...
        yc       (y, x) float64 ...
    Dimensions without coordinates: y, x
    Data variables:
        Tair     (y, x) float64 ...
    🔑 bar
    <xarray.Dataset>
    Dimensions:  (time: 36, x: 275)
    Coordinates:
    * time     (time) object 1980-09-16 12:00:00 ... 1983-08-17 00:00:00
        xc       (x) float64 ...
        yc       (x) float64 ...
    Dimensions without coordinates: x
    Data variables:
        Tair     (time, x) float64 ...
    """

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Collection):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for key in sorted(self.keys()):
            try:
                xr.testing.assert_identical(self[key], other[key])
            except AssertionError:
                return False
        return True

    def __repr__(self) -> str:

        output = ''.join(f'{unicode_key} {key}\n{repr(value)}\n\n' for key, value in self.items())
        return f'<{type(self).__name__} ({len(self)} keys)>\n{output}'

    def _repr_html_(self):
        """
        Return an html representation for the collection object.
        Mainly for IPython notebook
        """

        def _summarize_datasets(datasets):
            ds_li = ''.join(
                f"<li class='xr-var-item'><strong>{unicode_key}&nbsp;{key}</strong>{xr.core.formatting_html.dataset_repr(ds)}</li>"
                for key, ds in datasets.items()
            )
            return f'<ul>{ds_li}</ul>'

        keys_section = functools.partial(
            xr.core.formatting_html._mapping_section,
            name='Datasets',
            details_func=_summarize_datasets,
            max_items_collapse=5,
            expand_option_name='display_expand_data_vars',
            enabled=True,
        )
        obj_type = f'xcollection.{type(self).__name__}'
        header = f"<div class='xr-header'><div class='xr-obj-type'>{escape(obj_type)}</div></div>"
        return (
            '<div>'
            "<div class='xr-wrap' style='display:none'>"
            f'{header}'
            f'{keys_section(self.datasets)}'
            '</div>'
            '</div>'
        )

    def _ipython_display_(self):
        """
        Display the collection in the IPython notebook.
        """
        from IPython.display import HTML, display

        display(HTML(self._repr_html_()))

    def keys(self) -> typing.Iterable[str]:
        """Return the keys of the collection."""
        return self.datasets.keys()

    def values(self) -> typing.Iterable[xr.Dataset]:
        """Return the values of the collection."""
        return self.datasets.values()

    def items(self) -> typing.Iterable[typing.Tuple[str, xr.Dataset]]:
        """Return the items of the collection."""
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

        Examples
        --------
        >>> c
        <Collection (3 keys)>
        🔑 foo
        <xarray.Dataset>
        Dimensions:  (y: 205, x: 275)
        Coordinates:
            time     object 1980-09-16 12:00:00
            xc       (y, x) float64 ...
            yc       (y, x) float64 ...
        Dimensions without coordinates: y, x
        Data variables:
            Tair     (y, x) float64 ...
        🔑 bar
        <xarray.Dataset>
        Dimensions:  (time: 36, x: 275)
        Coordinates:
        * time     (time) object 1980-09-16 12:00:00 ... 1983-08-17 00:00:00
            xc       (x) float64 ...
            yc       (x) float64 ...
        Dimensions without coordinates: x
        Data variables:
            Tair     (time, x) float64 ...
        🔑 baz
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            *empty*
        >>> len(c)
        3
        >>> c.keys()
        dict_keys(['foo', 'bar', 'baz'])
        >>> d = c.choose(data_vars=['Tair'], mode='any')
        >>> len(d)
        2
        >>> d.keys()
        dict_keys(['foo', 'bar'])
        >>> d = c.choose(data_vars=['Tair'], mode='all')
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

    def filter(self, *, by: str, func: typing.Callable) -> 'Collection':
        """Return a collection with datasets that match the filter function.

        Parameters
        ----------
        by : str
            Option to filter by. Must be one of 'key', 'value', or 'item'.
        func : callable
            The filter function.

        Returns
        -------
        Collection
            A new collection containing only the selected datasets.

        Examples
        --------
        >>> c
        <Collection (3 keys)>
        🔑 foo
        <xarray.Dataset>
        Dimensions:  (y: 205, x: 275)
        Coordinates:
            time     object 1980-09-16 12:00:00
            xc       (y, x) float64 ...
            yc       (y, x) float64 ...
        Dimensions without coordinates: y, x
        Data variables:
            Tair     (y, x) float64 ...
        🔑 bar
        <xarray.Dataset>
        Dimensions:  (time: 36, x: 275)
        Coordinates:
        * time     (time) object 1980-09-16 12:00:00 ... 1983-08-17 00:00:00
            xc       (x) float64 ...
            yc       (x) float64 ...
        Dimensions without coordinates: x
        Data variables:
            Tair     (time, x) float64 ...
        🔑 baz
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            *empty*
        >>> len(c)
        3
        >>> c.keys()
        dict_keys(['foo', 'bar', 'baz'])
        >>> c.filter(by='key', func=lambda key: isinstance(key, str))
        >>> c.filter(by='value', func=lambda ds: 'Tair' in ds.data_vars)
        >>> c.filter(
        ...     by='item',
        ...     func=lambda item: 2014 in item[1].time.dt.year and isinstance(item[0], str),
        ... )

        """

        _VALID_BY = ['key', 'value', 'item']
        if by not in _VALID_BY:
            raise ValueError(f'Invalid by: {by}. Accepted by are {_VALID_BY}')

        if by == 'key':
            result = toolz.keyfilter(func, self.datasets)

        elif by == 'value':
            result = toolz.valfilter(func, self.datasets)

        elif by == 'item':
            result = toolz.itemfilter(func, self.datasets)

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

        Examples
        --------
        >>> c
        <Collection (2 keys)>
        🔑 foo
        <xarray.Dataset>
        Dimensions:  (y: 205, x: 275)
        Coordinates:
            time     object 1980-09-16 12:00:00
            xc       (y, x) float64 ...
            yc       (y, x) float64 ...
        Dimensions without coordinates: y, x
        Data variables:
            Tair     (y, x) float64 ...
        🔑 bar
        <xarray.Dataset>
        Dimensions:  (time: 36, x: 275)
        Coordinates:
        * time     (time) object 1980-09-16 12:00:00 ... 1983-08-17 00:00:00
            xc       (x) float64 ...
            yc       (x) float64 ...
        Dimensions without coordinates: x
        Data variables:
            Tair     (time, x) float64 ...
        >>> c.keys()
        dict_keys(['foo', 'bar'])
        >>> d = c.keymap(lambda x: x.upper())
        >>> d.keys()
        dict_keys(['FOO', 'BAR'])

        """
        if not callable(func):
            raise TypeError(f'First argument must be callable function, got {type(func)}')

        return type(self)(datasets=toolz.keymap(func, self.datasets))

    def map(
        self,
        func: typing.Callable[[xr.Dataset], xr.Dataset],
        args: typing.Sequence[typing.Any] = None,
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

        Examples
        --------
        >>> c
        <Collection (2 keys)>
        🔑 foo
        <xarray.Dataset>
        Dimensions:  (y: 205, x: 275)
        Coordinates:
            time     object 1980-09-16 12:00:00
            xc       (y, x) float64 ...
            yc       (y, x) float64 ...
        Dimensions without coordinates: y, x
        Data variables:
            Tair     (y, x) float64 ...
        🔑 bar
        <xarray.Dataset>
        Dimensions:  (time: 36, x: 275)
        Coordinates:
        * time     (time) object 1980-09-16 12:00:00 ... 1983-08-17 00:00:00
            xc       (x) float64 ...
            yc       (x) float64 ...
        Dimensions without coordinates: x
        Data variables:
            Tair     (time, x) float64 ...
        >>> c.map(func=lambda x: x.isel(x=slice(0, 10)))
        <Collection (2 keys)>
        🔑 foo
        <xarray.Dataset>
        Dimensions:  (y: 205, x: 10)
        Coordinates:
            time     object 1980-09-16 12:00:00
            xc       (y, x) float64 ...
            yc       (y, x) float64 ...
        Dimensions without coordinates: y, x
        Data variables:
            Tair     (y, x) float64 ...
        🔑 bar
        <xarray.Dataset>
        Dimensions:  (time: 36, x: 10)
        Coordinates:
        * time     (time) object 1980-09-16 12:00:00 ... 1983-08-17 00:00:00
            xc       (x) float64 ...
            yc       (x) float64 ...
        Dimensions without coordinates: x
        Data variables:
            Tair     (time, x) float64 ...
        """
        args = args or ()

        if not callable(func):
            raise TypeError(f'First argument must be callable function, got {type(func)}')

        if not isinstance(args, tuple):
            raise TypeError(f'Second argument must be a tuple, got {type(args)}')

        func = _rpartial(func, *args, **kwargs)
        return type(self)(datasets=toolz.valmap(func, self.datasets))

    def to_zarr(self, store, mode: str = 'w', **kwargs):
        """Write the collection to a Zarr store.

        Parameters
        ----------
        store : str or pathlib.Path
             Store or path to directory in local or remote file system.
        mode : {"w", "w-", "a", "r+", None}, optional
            Persistence mode: "w" means create (overwrite if exists);
            "w-" means create (fail if exists);
            "a" means override existing variables (create if does not exist);
            "r+" means modify existing array *values* only (raise an error if
            any metadata or shapes would change).
            The default mode is "a" if ``append_dim`` is set. Otherwise, it is
            "r+" if ``region`` is set and ``w-`` otherwise.
        kwargs
            Additional keyword arguments to pass to :py:meth:`~xarray.Dataset.to_zarr` method.

        Examples
        --------
        >>> c.to_zarr(store='/tmp/foo.zarr', mode='w')
        """

        if kwargs.get('group', None) is not None:
            raise NotImplementedError(
                'specifying a root group for the collection has not been implemented.'
            )

        return [value.to_zarr(store, group=key, mode=mode, **kwargs) for key, value in self.items()]

    def weighted(self, weights, **kwargs) -> 'Collection':
        """Return a collection with datasets weighted by the given weights."""
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


def open_collection(store: typing.Union[str, pydantic.DirectoryPath], **kwargs):
    """Open a collection stored in a Zarr store.

    Parameters
    ----------
    store : str or pathlib.Path
         Store or path to directory in local or remote file system.
    kwargs
        Additional keyword arguments to pass to :py:func:`~xarray.open_dataset` function.

    Returns
    -------
    Collection
        A collection containing the datasets in the Zarr store.

    Examples
    --------
    >>> import xcollection as xc
    >>> c = xc.open_collection('/tmp/foo.zarr', decode_times=True, use_cftime=True)

    """

    import zarr

    zstore = zarr.open_group(store, mode='r')
    datasets = {
        key: xr.open_dataset(store, group=key, engine='zarr', **kwargs)
        for key in zstore.group_keys()
    }
    return Collection(datasets=datasets)
