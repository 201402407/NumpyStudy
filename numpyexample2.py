#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import scipy as sp

def pprint(arr):
    print("type : {}".format(type(arr)))
    print("shape : {}, dimension : {}, dtype : {}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array Data : \n", arr)
# In[ ]:
# example
arr = np.array([[[1,2,3], [4,5,6]], [[3,2,1], [4,5,6]]], dtype = float)
a= np.array(arr, dtype = float)
pprint(arr)
# In[ ]:

