---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Tutorial

xcollection extends [xarray's data model](https://xarray.pydata.org/en/stable/getting-started-guide/why-xarray.html) to be able to handle a dictionary of xarray Datasets. A {py:class}`xcollection.main.Collection` behaves like a regular dictionary, but it also has a few extra methods that make it easier to work with.

Let's start by importing the necessary packages.

```{code-cell} ipython3
import xarray as xr
import xcollection as xc
import typing
```

## Creating a collection from a dictionary of datasets

To create a collection, we just pass a dictionary of {py:class} `xarray.Dataset` to the {py:class}`xcollection.main.Collection` constructor.

```{code-cell} ipython3
ds = xr.tutorial.open_dataset('air_temperature')
ds.attrs = {}
dsa = xr.tutorial.open_dataset('rasm')
dsa.attrs = {}
```

```{code-cell} ipython3
col = xc.Collection({'foo': ds, 'bar': dsa})
col
```

## Accessing keys and values in a collection

To access the keys and values of a collection, we can use the {py:func}`xcollection.main.Collection.keys` and {py:func}`xcollection.main.Collection.values` methods.

```{code-cell} ipython3
col.keys()
```

```{code-cell} ipython3
col.values()
```

In addition, we can use the {py:func}`xcollection.main.Collection.items` method to get a list of tuples of the keys and values.

```{code-cell} ipython3
for key, value in col.items():
    print(key, value)
```

## Mapping operations over a collection

xcollection provides a number of methods that allow us to map arbitrary operations (functions) over the keys and values of a collection.

One such method is {py:func}`xcollection.main.Collection.map`. This method takes a function and applies it to the values of the collection and returns a new collection.
To demonstrate this, we'll create a new collection with the same keys and values as the original, but with the values of the original collection subsetted along the time dimension.

```{code-cell} ipython3
def subset(ds: xr.Dataset, dim_slice: typing.Dict[str, slice]):
    return ds.isel(**dim_slice)


new_col = col.map(subset, dim_slice={"time": slice(0, 3)})
new_col
```

As you can see, the new collection has the same keys as the original, but the values are subsets of the original values.

Another method is {py:func}`xcollection.main.Collection.keymap`. This method takes a function and applies it to the **keys** of the collection and returns a new collection. This is useful for manipulating the keys of the collection. Let's create a new collection with the same keys as the original, but with the keys capitalized.

```{code-cell} ipython3
def capitalize(key: str):
    return key.upper()

new_col_capitalized = col.keymap(capitalize)
new_col_capitalized
```

## Filtering a collection

xcollection provides a number of methods that allow us to filter the keys and values of a collection. One such method is {py:func}`xcollection.main.Collection.filter`. This method expectes two arguments:

1. `func`: a function that returns a boolean value
2. `by`: which specifies whether the function is applied on keys, values or items.

### Filtering based on keys

```{code-cell} ipython3
def contains_foo(key: str) -> bool:
    return 'foo' in  key.lower()

col.filter(func=contains_foo, by='key')
```

### Filtering based on values

```{code-cell} ipython3
def contains_time(ds: xr.Dataset) -> bool:
    return 'time' in ds.coords

col.filter(func=contains_time, by='value')
```

### Filtering based on items

```{code-cell} ipython3
def contains_foo_and_spans_2014(item: tuple) -> bool:
    key, ds = item
    return 'foo' in key.lower() and 2014 in ds.time.dt.year

col.filter(func=contains_foo_and_spans_2014, by='item')
```

## Choosing a subset of a collection based on data variables

xcollection provides a {py:func}`xcollection.main.Collection.choose` method that allows us to filter a collection based on whether datasets in a collection contain one or more data variables. For example, our existing `col` collection contains two datasets. `foo` consists of a dataset with `air` as a data variable and `bar` has `air_temperature` as a data variable. We can filter the collection to only include datasets that have `air` as a data variable as follows:

```{code-cell} ipython3
new_col = col.choose(['air'], mode='any')
new_col
```

As you can see in the output, the new collection only contains the dataset `foo`.

By default, the `mode` argument is set to `any`, meaning that the collection will only contain datasets that contain one or more data variables. If we set the `mode` argument to `all`, xcollection will error if any of the datasets in the collection do not contain all of the data variables specified.

```{code-cell} ipython3
new_col = col.choose(['air'], mode='all')
```

As you can see in the output, we get error because the dataset `bar` does not contain the `air` data variable.

## Saving a collection to disk

To save a collection to disk, we can use the {py:func}`xcollection.main.Collection.to_zarr` method. This method takes a path to a directory or a cloud bucket storage and writes the collection as a zarr store. Each key in the collection is saved as a zarr group with the same name as the key.

```{code-cell} ipython3
col.to_zarr('/tmp/my_collection.zarr', consolidated=True, mode='w')
```

```{code-cell} ipython3
!ls -ltrha /tmp/my_collection.zarr
```

## Loading a collection from disk

To load a collection from disk, xcollection provides a {py:func}`xcollection.main.open_collection` function. This method takes a path to a directory or a cloud bucket storage and reads the collection from a zarr store.

```{code-cell} ipython3
new_col = xc.open_collection('/tmp/my_collection.zarr')
assert col == new_col
new_col
```
