from hoge import Hoge
import pytest

import inspect
import re
import sys


def getNotTestFunctions(package):
    members = inspect.getmembers(package)
    members = filter((lambda m: inspect.isfunction(m[1]) ),members)
    v = []
    for (name,ptr) in members:
        s = inspect.getfullargspec(ptr)
        if ptr.__doc__ == None or \
           ( ptr.__doc__.count('shape_eq') == 0 or \
             ptr.__doc__.count('inout_eq') == 0 or \
             ptr.__doc__.count('inout_eq') == 0 ) :
          v.append(name)
    return v

def test_hoge():
  obj = Hoge.Piyo()
  assert 1 == obj.one()


def test_detect_not_tested_functions():
  not_tested_functions = [
    "inout_eq",
    "prop_int_eq",
    "shape_eq",
    "ttype_eq",
    "type_eq"
  ]
  assert getNotTestFunctions(Hoge) == not_tested_functions
