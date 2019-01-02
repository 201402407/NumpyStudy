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
# else
import timeit

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
# series 리스트 생성
s = pd.Series([1, 3, 5, np.nan, 6, 8])
pprint(s)
print(s.values)
print(s.index)
# In[ ]:
# 원하는 index 주기
s = pd.Series([1, 5, 8, 3, 10, 22], index=["처음", "둘", "셋", "넷", "데섯", "마지막"])
print(s)
# In[ ]:
# 인덱스, 인덱스의 이름, series의 이름 변경 가능
s.index.name = "내맘"
s.name = "튜토리얼"
print(s)
s.index = ["11", "22", "33", "44", "55", "66"]
print(s)
# In[ ]:
# dataFrame 생성
datas = pd.date_range('20181231', periods=6)    # 시작 날짜와 그 기간을 dictionary 형태로 리턴하는 함수
# index는 사용하려는 인덱스 배열, columns는 list를 인자로 받아 각 열의 제목으로 만든다
df = pd.DataFrame(np.random.randint(1, 10, size=(6, 4)), index=datas, columns=list('가나다라'))
df
# In[ ]:
# index, columns, values 확인
print(df.index)
print(df.columns)
df.values
# In[ ]:
# dataFrame 값
df
# In[ ]:
# describe() 함수 -> 통계량 ( 총 개수, 평균, 표준편차, 최소값, 퍼센트, 최대값 )
df.describe()
# In[ ]:
# index와 column 바꾼 값
df.T
# In[ ]:
# 인자로 받은 조건에 따른 정렬
df.sort_index(axis=1, ascending=False) # axis = 1 : columns 기준. ascending = False : 내림차순
# In[ ]:
# 내부 값 정렬
df.sort_values(by = "다") # "다" 열의 값들 전부 오름차순

# In[ ]:
# 하나의 열 선택
# df["열 이름"], df.열이름