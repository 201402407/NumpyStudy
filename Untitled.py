#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import timeit


# In[2]:


# 1차원 배열
A = np.array([1,2,3,4]) # np.array()는 배열을 생성하는 함수
print("A.ndim : ", A.ndim) # np.ndim은 배열의 차원 개수 리턴
print("A.reshape : ", A.shape) # np.shape는 배열의 각 차원의 크기 튜플로 리턴. 여기서는 첫 번째 차원의 크기만 출력됨
print("A : ", A)


# In[3]:


# 2차원 배열
B = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]) # 2차원으로 생성
print("B.ndim : ", B.ndim) # np.ndim은 배열의 차원 개수 리턴
# np.shape는 배열의 각 차원의 크기 튜플로 리턴. 여기서는 첫 번째 차원의 크기, 두번째 차원의 크기만 출력됨
# 즉, 첫 번째 : 열, 두 번째 : 행
print("B.reshape : ", B.shape) 
print("B : ", B)


# In[4]:


# 3차원 배열
C = np.array([ [[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]  ]) # 3차원으로 생성
print("C.ndim : ", C.ndim) # np.ndim은 배열의 차원 개수 리턴
# np.shape는 배열의 각 차원의 크기 튜플로 리턴. 여기서는 총 세 개의 값만 출력됨
# 즉, 첫 번째 : 2차원 배열 크기, 두 번째 : 하나의 2차원 배열의 열, 세 번째: 행
print("C.reshape : ", C.shape) 
print("C : ", C)


# In[ ]:

# for문과 shape를 이용한 index 찾기
for x in range(C.shape[0]): # 2차원 배열 두 개를 가지고 for문 실행
    for y in range(C.shape[1]):  # 위에서 정한 2차원 배열의 열 선택
        for z in range(C.shape[2]): # 2차원 배열의 행 선택
            print("C[%d, %d, %d] = %d   " % (x, y, z, C[x,y,z]), end="\t") # 각각의 x,y,z는 index가 되고, end="" 는 print 문의 맨 끝에 추가할 거 적는것.
        print("\n")

# In[ ]:

# zeros, ones, empty, full, eye, random

# zeros
A = np.zeros((3,4)) # shape가 3,4인 배열을 생성하고 0으로 채움
print("A = \n", A)
print("A.dtype = ", A.dtype) # 각 원소의 type. 따로 지정하지 않으면 float64

# In[ ]:

# ones
A = np.ones((3,4), dtype=np.int32) # shape가 3,4인 배열을 생성하고 1로 채움. type는 int32로 설정.
print("A = \n", A)
print("A.dtype = ", A.dtype) # 위에서 변경했으므로 int32

# In[ ]:

# empty
A = np.empty((4,4)) # empty는 초기값을 설정하지 않기 때문에 임의의 값이 나옴. 장점은 생성속도가 빠름
print("A = \n", A)
print("A.dtype = ", A.dtype)

# In[ ]:

# full
A = np.full((3,3), 33) # full은 shape만큼 배열을 생성한 다음 원소를 입력한 값으로 설정
print("A = \n", A)
print("A.dtype = ", A.dtype)

# In[ ]:

# eye
A = np.eye(4) # eye는 n*n 단위행렬 생성
print("A = \n", A)
print("A.dtype = ", A.dtype)

# In[ ]:

# random
A = np.random.random((2,2)) # random은 각각의 값이 전부 랜덤으로 채워짐
print("A = \n", A)
print("A.dtype = ", A.dtype)

# In[ ]:

# arange(시작값, 마지막값, 간격)
A = np.arange(0, 2, 0.2) # random은 각각의 값이 전부 랜덤으로 채워짐
print("A = \n", A)
print("A.dtype = ", A.dtype)

# In[ ]:

# linspace(시작값, 마지막값, 샘플 개수)
# 시작값 <= x <= 마지막값 범위 내에서 일정 간격으로 지정한 개수만큼 배열 생성
A = np.linspace(0, 10, 10) 
print("A = \n", A)
print("원소간 간격 = {}".format(10/9)) # 원소간 간격은 마지막값/(샘플개수 - 1)
print("A.dtype = ", A.dtype)
print("A.size = ", A.size)

# In[ ]:

# reshape
A = np.arange(16)
B = A.reshape(4, 4) # ndarray A의 데이터를 공유하지만 A와는 다른 shape()를 가지고 생성

print("A = \n", A)
print("B = {} \n".format(B))
print("----------------------------------\n", end="\n")
# In[ ]:
# 특징 1
# 다른 배열의 데이터를 공유하고 있는지 여부 확인
print("A.base = \n", A.base)
print("B.base = {} \n".format(B.base))
print("----------------------------------\n", end="\n")
# In[ ]:
# 특징 2
# 두 배열이 동일한 객체인지 확인
if B.base is A:
    print("두 객체가 같음. \n", end="\n")
else:
    print("같지않음.", end="\n")
print("----------------------------------\n", end="\n")
# In[ ]:
# 특징 3
# 어느 한 쪽을 변경하면 같이 변경됨
B[0][1] = 55
print("A = \n", A)
print("B = {} \n".format(B))
print("----------------------------------\n", end="\n")
# In[ ]:
# 특징 3-1
# copy와 reshape는 공유 유무의 차이임 -> copy는 공유하지 않으므로 B의 데이터가 변하지 않음
C = B.reshape(2, 8).copy()
print("C = \n", C)
print("C.base = {} \n".format(C.base))
C[0] = 44
print("B = \n", B)
print("C = {} \n".format(C))
print("----------------------------------\n", end="\n")
# In[ ]:
# 특징 4
# 차원 하나를 -1로 적으면(shape) 배열 전체 원소 개수와 확정된 차원 크기로부터 남은 차원의 크기를 추론하여 배열 생성
D = C.reshape(4, -1)  
print("D = \n", D)
print("D.shape = \n", D.shape)
print("----------------------------------\n", end="\n")

# In[ ]:
# ravel은 배열의 모든 데이터를 1차원 행렬로 변환해서 리턴
E = D.ravel()
print("E = {} \n".format(E))
print("----------------------------------\n", end="\n")

# In[ ]:
# resize 실행 시 공유상태이면 공유시작한 배열한테는 영향을 주지 않음
A = E.reshape(4, 4)
print("바뀌기 전 A = \n", A)
A.resize(8, 2)
print("바뀌고 난 뒤 A = \n", A)
print("바뀌기 전 E = \n", E)
print("----------------------------------\n", end="\n")
# In[ ]:
# numpy.vstack은 두 배열을 세로 방향으로 연결해서 하나의 큰 배열로 만듬
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 0], [0, 1]])
C = np.vstack((A,B))
print("C = np.vstack(A,B)\n", np.vstack((A,B)))
print("----------------------------------\n", end="\n")
# numpy.hstack은 두 배열을 가로 방향으로 연결해서 하나의 큰 배열로 만듬
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 0], [0, 1]])
C = np.hstack((A,B))
print("C = np.hstack(A,B)\n", np.hstack((A,B)))
print("----------------------------------\n", end="\n")

# In[ ]:
# numpy.column_stack은 1차원 배열들을 가지고 2차원 배열로 새로 생성함
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = [9, 10, 11, 12]
E = np.column_stack((a,b,c))
E[:] = 0         
print("E = \n", E)
print("a = ", a)  
print("b = ", b)
print("c = ", c)
print("----------------------------------\n", end="\n")

# In[ ]:

# indexing 
# 1차원 배열
A = np.arange(1, 11, 2)
print("A = \n", A)
for index, a in enumerate(A): # 배열의 index와 element를 묶어 반환. enumerate(배열 name). a : element
    print("A[%d] =" % (index), a, end=", ")
    print(a.dtype)
print("----------------------------------\n", end="\n")
# In[ ]:
# 1차원 배열 인덱스 역순 
for index, a in zip(range(0,-(len(A)), -1), A): # zip(range(), 배열 name)
    print("A[%d]=" % (index), a, end=", ")
    print(a.dtype)
print("----------------------------------\n", end="\n")\
# In[ ]:
# slising
# 1차원 배열 -> slising은 기존 파이썬의 slising과 유사
A = np.arange(1, 11, 2)

print("A[0:3]=", A[0:3])  # 슬라이스 A[0:3]의 원소는 A[0], A[1], A[2]으로 구성됩니다.
print("A[:5]=", A[:5])    # 시작 인덱스는 명시하지 않은 것이랑
print("A[6:]=", A[6:]) # 끝 인덱스를 명시하지 않으면 마지막 인덱스도 포함되지만

print("A[:]=", A[:]) # 배열의 전체 원소를 포함하는 슬라이스입니다.
print("A[::2]=", A[::2]) # 첫번째 원소부터 2씩 건너띄며 원소를 취합니다. 배열의 마지막 원소까지를 대상으로 합니다.
print("A[3::2]=", A[3::2]) # 네번째 원소부터 2씩 건너띄며 원소를 취합니다. 배열의 마지막 원소까지를 대상으로 합니다.
print("\n")
# In[ ]:
# 하나의 배열을 특정 인덱스를 기준으로 둘로 분리해봅니다.  
print("A[:-2]=", A[:-2]) # 마지막 두 개의 원소를 제외하고 취합니다.
print("A[-2:]=", A[-2:]) # 마지막 두 개만 취합니다.
print("\n")

print("변경 전 A=", A)
A[0:3]=100  # 배열의 슬라이스를 변경하면
print("슬라이스 값 변경 후 A[0:3]=", A[0:3]) #슬라이스의 모든 원소값 뿐만아니라
print("슬라이스 값 변경 후 A=", A) #원본 배열에서 해당 부분의 데이터가 변경됩니다.
print("\n")
# In[ ]:
# 하나의 배열에 대한 모든 슬라이스의 id는 모두 동일합니다.
# 하지만 배열 A의 id와는 다릅니다.
print("id(A)=", id(A), " ", A.size)
print("id(A[:])=", id(A[:]), " ", A[:].size)
print("id(A[0:3])=", id(A[0:3]), " ", A[0:3].size)
print("id(A[:-1])=", id(A[0:-1]), " ", A[0:-1].size)
print("\n")

# 인덱스로 참조한 배열의 원소의 id는 모두 동일합니다.
# 하지만 배열 A의 id와는 다릅니다.
print("id(A)=", id(A), " ", A.size)
print("id(A[0])=", id(A[0]), " ", A[0].size)
print("id(A[1])=", id(A[1]), " ", A[1].size)
print("id(A[-1])=", id(A[-1]), " ", A[-1].size)
print("\n")

# In[ ]:
# 2차원 배열
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# indexing & slising
print("A = \n", A)
print("A[1, 2] = ", A[1, 2])  # 두번째 행, 세번째 열의 원소의 값을 출력.
print("A[1][2] = ", A[1][2])  # 위와 동일한 결과를 출력.
# 하나의 행을 슬라이싱
print("A[0]=", A[0])
# 하나의 열을 슬라이싱
print("A[:, 1] = ", A[:, 1])   # 전체 행( : ) 중에서 두번째 열( 1 )에 해당되는 원소

# 배열의 일부를 슬라이싱해서 값을 변경하면 원본 배열의 값도 변경됨.
# In[ ]:
for row in A:   # 배열의 행벡터 출력
    print(row)
print("\n")
for column in A.T:   # A.T는 행과 열을 교환하여 얻게되는 전치 행렬(transpose matrix).
    print(column)
print("\n")

# In[ ]:
# 같은 크기의 연산
A = np.array([2, 4, 6, 8]).reshape(2,2)
B = np.array([2, 2, 2, 2]).reshape(2,2)
print("A = \n", A)
print("B = \n", B)

print("A+B =\n", A+B)
print("A-B =\n", A-B)
print("A*B(np.matmul(A,B)) =\n", np.matmul(A,B))
print("A/B =\n", A/B)