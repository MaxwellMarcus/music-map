import numpy as np

def augment( data, transformation=None ):
    return { "data": transformation @ data }