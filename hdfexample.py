#!/usr/bin/env python
# coding: utf-8

# In[1]:
# numpy
import numpy as np
# scipy
import scipy as sp
import scipy.stats
# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix
from scipy import linalg
# pandas
import pandas as pd
# netcdf
from netCDF4 import Dataset
# else
import timeit
import time

# In[2]:
# 기본적인 세팅
def pprint(arr):
    print("type : {}".format(type(arr)))
    print("shape : {}, dimension : {}, dtype : {}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array Data : \n", arr)

print ('버전: ', mpl.__version__)
print ('설치 위치: ', mpl.__file__)
print ('설정 위치: ', mpl.get_configdir())
print ('캐시 위치: ', mpl.get_cachedir())
print ('설정 파일 위치: ', mpl.matplotlib_fname())
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# 해당 파일 목록 중 Nanum이 들어간 글꼴 경로 찾기
[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]

# path에 경로 적고 plt.rc로 폰트 설정
path = 'C:\\Windows\\Fonts\\NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=path, size=40).get_name()
print(font_name)
plt.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False  # 그래프 안에 '-' 부호도 표시되도록 하는 함수

# In[ ]:
# hdf 파일 불러오기
