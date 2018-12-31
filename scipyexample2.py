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
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix
from scipy import linalg
# else
import timeit

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
mpl.rcParams['axes.unicode_minus'] = False

# In[1]:
# 가우시안 정규 분포 객체 생성
rv = sp.stats.norm(loc=10, scale=10) # loc(평균)는 10, scale(표준편차)는 10
#rv.rvs(size=(3,10), random_state=1) # size(shape)는 (3,10)이고 random_state(seed)는 1

# In[2]:
# pdf 예시
xx = np.linspace(-10, 10, 100)
pdf = rv.pdf(xx)
plt.plot(xx, pdf)               # xx는 x, pdf는 y 좌표 값을 가리킴
plt.title("확률밀도함수 ")      # 제목(맨 위의 가운데)
plt.xlabel("x의 값")              # x축 좌표 이름
plt.ylabel("p(x) 값")           # y축 좌표 이름
plt.show()

# In[ ]:
# cdf 예시
xx = np.linspace(-10, 10, 100)
cdf = rv.cdf(xx)
plt.plot(xx, cdf)
plt.title("누적분포함수 ")
plt.xlabel("x의 값")
plt.ylabel("F(x) 값")
plt.show()