---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Tutorial

```{code-cell} ipython3
import xarray as xr
import xcollection as xc
```

## Creating a collection from a dictionary of datasets

A {py:class}`xcollection.main.Collection` is a container for a dictionary of {py:class}`xarray.Dataset` objects.

```{code-cell} ipython3
ds = xr.tutorial.open_dataset('air_temperature')
dsa = xr.tutorial.open_dataset('rasm')
```

```{code-cell} ipython3
col = xc.Collection({'foo': ds, 'bar': dsa})
print(col)
```
