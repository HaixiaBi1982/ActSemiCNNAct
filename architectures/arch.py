#!coding:utf-8
from functools import wraps

from architectures.convlarge1D import convLarge1D

arch = {
        'cnn13_1D': convLarge1D
        }


def RegisterArch(arch_name):
    """Register a model
    you must import the file where using this decorator
    for register the model function
    """
    def warpper(f):
        arch[arch_name] = f
        return f
    return warpper
