from argparse import Namespace
from types import SimpleNamespace
from collections import OrderedDict
from avssl.base import OrderedNamespace


def test_dict():
    # Basic tests
    d_1 = {
        "a": 1,
        "b": [2, {"c": 3}],
        "d": {"e": 4, "f": "g"},
        "h": SimpleNamespace(i=5, j={"k": 6}),
    }
    ons_1 = OrderedNamespace(d_1)
    ons_2 = OrderedNamespace(**d_1)

    assert ons_1.a == ons_1["a"]
    assert ons_1.b == ons_1["b"]
    assert ons_1.b[0] == d_1["b"][0]
    assert ons_1.b[1].c == d_1["b"][1]["c"]
    assert ons_1.d.e == ons_1["d"]["e"]
    assert ons_1.d["e"] == ons_1["d"]["e"]
    assert ons_1.h.i == 5
    assert ons_1.h.j.k == 6
    assert ons_1 == ons_2
    assert len(ons_1) == 4
    assert len(ons_1.keys()) == len(ons_1)
    assert "a" in ons_1

    d_2 = ons_1.pydict
    d_2_2 = ons_1.to_dict()
    od_1 = ons_1.odict
    od_1_2 = ons_1.to_odict()

    assert ons_1.keys() == d_2.keys()
    assert ons_1.keys() == od_1.keys()
    assert d_2 == d_2_2
    assert od_1 == od_1_2
    assert isinstance(d_2, dict)
    assert isinstance(d_2_2, dict)
    assert isinstance(od_1, OrderedDict)
    assert isinstance(od_1_2, OrderedDict)

    # Consistency among dict and namespaces
    d_3 = {"a": 1, "b": 2}
    ns_1 = SimpleNamespace(a=1, b=2)
    ns_2 = Namespace(a=1, b=2)

    ons_3 = OrderedNamespace(d_3)
    ons_4 = OrderedNamespace(ns_1)
    ons_5 = OrderedNamespace(ns_2)

    assert ons_3 == ons_4
    assert ons_3 == ons_5
    assert ons_4 == ons_5

    # Merge dict and namespace
    d_4 = {"a": 1, "b": 2, "c": {"d": 3}}
    ns_3 = Namespace(e=4, f=SimpleNamespace(g=5))
    ons_6 = OrderedNamespace([d_4, ns_3])

    assert ons_6.a == 1
    assert ons_6.c.d == 3
    assert ons_6.e == 4
    assert ons_6.f.g == 5
    assert ons_6.f["g"] == 5
