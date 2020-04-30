from pandas.core.internals import SingleBlockManager
from shapely.geometry import Point, Polygon
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas import Series, DataFrame, array
import numpy as np
import geopandas as gpd
from pandas.api.extensions import register_extension_dtype
from arctern import *
import pyarrow


@register_extension_dtype
class MyDtype(ExtensionDtype):
    type = bytes
    name = "mydtype"  # for series.dtype
    na_value = None  # maybe invalid gis value
    kind = 'O'

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return MyArry


class MyArry(ExtensionArray):
    _dtype = MyDtype()

    def __init__(self, data):
        # data is numpy.arry
        self.data = data

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        # print("getitem type: ", type(item))
        return item

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        return MyArry(result)

    def copy(self):
        return MyArry(self.data.copy())

    def _formatter(self, boxed=False):
        return lambda x: ST_AsText([x])[0]

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return len(self.shape)

    def isna(self):
        return np.array([g is None for g in self.data], dtype="bool")

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return MyArry(from_np_arry(scalars))

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        return pyarrow.array(self.data, type=type)

    def __eq__(self, other):
        # return np.array([True] * len(self))
        r = ST_Equals(self.data, ST_GeomFromText(Series(other))).values
        return r


def from_np_arry(data):
    out = ST_GeomFromText(data).values
    return out


class MySeries(Series):
    _metadata = ["name"]

    def __new__(cls, data=None, index=None, **kwargs):
        name = kwargs.pop("name", None)

        if isinstance(data, bytes):
            n = len(index) if index is not None else 1
            data = [data] * n
        else:
            s = Series(data, index=index, name=name, **kwargs)
            data = MyArry(from_np_arry(s.values))
            index = s.index
        self = super(MySeries, cls).__new__(cls)
        super(MySeries, self).__init__(data, index=index, name=name, **kwargs)
        return self

    def __init__(self, *args, **kwargs):
        pass

    def func(self):
        return str("My series func")

    def __getitem__(self, key):
        val = getattr(super(MySeries, self), "__getitem__")(key)
        return val

    # TODO: could be delegated to MyArry.st_length
    @property
    def st_length(self):
        return ST_Length(self.values)

    @property
    def is_valid(self):
        return ST_IsValid(self.values)


if __name__ == "__main__":
    s = MySeries(['LINESTRING(0 0,0 1)', 'Point(1.0  9.9)'], name='test', index=['0', '1'])
    print(s)
    print("-" * 30)
    print("s.is_valid is :\n", s.is_valid)


    def my_union_agg(s):
        # TODO: actually we expect MyDtype data
        return ST_Union_Aggr(s)[0]


    r = s.groupby(s == 'LINESTRING(0 0,0 1)').agg(my_union_agg)
    print("-" * 30)
    print(r)

    print("-" * 30)
    c = s.groupby(s == 'LINESTRING(0 0,0 1)').agg('count')
    print(c)

