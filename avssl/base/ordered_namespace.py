from argparse import Namespace
from collections import OrderedDict
from types import SimpleNamespace
from typing import List, Union


class OrderedNamespace(object):
    # ref: https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
    # ref: https://stackoverflow.com/a/14048352
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return OrderedNamespace(**entry)
        return entry

    def set_odict(self, data):
        for key, val in data.items():
            if isinstance(val, (dict, OrderedDict)):
                self._odict[key] = OrderedNamespace(**val)
            elif isinstance(val, list):
                self._odict[key] = list(map(self.map_entry, val))
            elif isinstance(val, (SimpleNamespace, Namespace)):
                self._odict[key] = OrderedNamespace(val)

    def __init__(
        self,
        data: Union[
            dict,
            OrderedDict,
            SimpleNamespace,
            Namespace,
            list,
        ] = None,
        **kwargs
    ):
        """Ordered Namespace

        Args:
            data (Union[ dict, OrderedDict, SimpleNamespace, Namespace,
                List[dict, OrderedDict, SimpleNamespace, Namespace]], optional):
                Data to initialize.
                If data is a dict, OrderedDict, SimpleNamespace, or Namespace, the object
                is initialized by it.
                If data is a list, all its elements are merged together sequentially.
                Defaults to None.
        """
        if isinstance(data, (SimpleNamespace, Namespace)):
            super().__setattr__("_odict", OrderedDict(vars(data)))
            self.set_odict(vars(data))
        elif isinstance(data, (dict, OrderedDict)):
            super().__setattr__("_odict", OrderedDict(**data))
            self.set_odict(data)
        elif isinstance(data, (tuple, list)):
            super().__setattr__("_odict", OrderedDict())
            for d in data:
                if isinstance(d, (SimpleNamespace, Namespace)):
                    d = vars(d)
                self._odict.update(d)
                self.set_odict(d)
        else:
            super().__setattr__("_odict", OrderedDict(**kwargs))
            self.set_odict(kwargs)

    def __getattr__(self, key):
        odict = super().__getattribute__("_odict")
        if key in odict:
            return odict[key]
        return super(OrderedNamespace, self).__getattribute__(key)

    def __setattr__(self, key, val):
        self._odict[key] = val

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, val):
        self._odict[key] = val

    def __iter__(self):
        return self.pydict.__iter__()

    def __next__(self):
        return self.pydict.__next__()

    @property
    def __dict__(self):
        return self._odict

    def __getstate__(self):
        # print("__getstate__")
        return self.__dict__

    def __setstate__(self, state):
        # print("__setstate__")
        # print(state)
        super(OrderedNamespace, self).__setattr__("_odict", OrderedDict())
        self._odict.update(state)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._odict)

    def to_odict(self):
        out_odict = OrderedDict()
        for key, val in self._odict.items():
            if isinstance(val, OrderedNamespace):
                out_odict[key] = val.to_odict()
            else:
                out_odict[key] = val
        return out_odict

    def to_dict(self):
        out_dict = dict()
        for key, val in self._odict.items():
            if isinstance(val, OrderedNamespace):
                out_dict[key] = val.to_dict()
            else:
                out_dict[key] = val
        return out_dict

    @property
    def odict(self):
        return self.to_odict()

    @property
    def pydict(self):
        return self.to_dict()

    def __str__(self):
        return "OrderedNamespace(" + self.to_dict().__str__() + ")"

    def keys(self):
        return self._odict.keys()

    def items(self):
        return self._odict.items()

    def values(self):
        return self._odict.values()

    def copy(self):
        return self.__class__(self)

    def get(self, key, value=None):
        return self._odict.get(key, value)

    def __delitem__(self, key, dict_delitem=dict.__delitem__):
        self._odict.__delitem__(key, dict_delitem=dict.__delitem__)
