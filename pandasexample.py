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
df.가
# In[ ]:
df["가"]
# In[ ]:
# 두 개 이상의 열 선택 -> 대괄호 2개
df[["가", "라"]]
# In[ ]:
# 열의 값 일괄변경
df["가"] = 25
df["가"]
# 열의 값 따로변경
df["나"] = [1, 5, 6, 7, 8, 10]
df["나"]
# In[ ]:
# 새로운 열 생성 및 대입
df["갹"] = np.arange(10, 16) # 임의의 순서값 대입
df[0:2]
# In[ ]:
# 새로운 열 입력 시 series 대입
temp = pd.Series([2, 4, 5], index=['2019-01-01', '2019-01-03', '2019-01-05'])
df["예"]  = temp
df
# In[ ]:
# 행 선택 및 조작(loc)
df.index = ["one", "two", "three", "four", "five", "six"]
df
# In[ ]:
df.loc["three" : "five"]    # 원하는 행의 index 선택하기
# In[ ]:
df.loc["three", "갹"]   # 원하는 행과 열을 동시 선택하기
# In[ ]:
df.loc["one" : "three", "나" : "다"] # 원하는 여러 개의 행과 열을 동시 선택
# In[ ]:
# 행을 새로 생성, 추가
df.loc["seven"] = [500, "hei", "nice", True, -20, False]
df.loc["seven"]
# In[ ]:
# 행과 열을 index 순서로 가져오기
df.iloc[4:6, 2:4]   # 5:6번째 행과 3:4번째 열의 같은 값을 가져오기 
# In[ ]:
# Boolean 인덱싱
df.iloc[0:6] > 1 # 1부터 6번째 행 까지 중 조건에 따른 결과 출력

# In[ ]:
# NaN 처리
# 기본 세팅
df2 = pd.DataFrame(np.random.randint(5, 9, (6 , 5)))    # 임의의 DataFrame 생성
df2.index = pd.date_range("20190101", periods = 6)
df2.columns = ["A", "B", "C", "D", "E"]
df2["F"] = [np.nan, 2, np.nan, 5, 8, np.nan]
df2
# In[ ]:
# NaN이 포함된 행 제거
df2.dropna(how = "any") # NaN이 하나라도 있으면 그 행 제거
# In[ ]:
df2.dropna(how = "all") # NaN이 모두 있는 행만 제거
# In[ ]:
df2.isnull()    # NaN인 성분만 True로 출력
# In[ ]:
# isnull()을 이용해서 NaN이 포함된 행만 뽑아내기
df2.loc[df2.isnull()["F"]]
# In[ ]:
df2.fillna(value = "hello") # NaN의 값을 원하는 값으로 변경
# In[ ]:
# 행, 열 삭제
# 두 개의 행 삭제 -> .drop(해당인덱스)
df2.drop([pd.to_datetime("20190102") , pd.to_datetime("20190105")])
# In[ ]:
# 두 개의 열 삭제 -> .drop(해당열, axis=1)
df2.drop(["A", "D"], axis=1)