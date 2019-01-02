#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import timeit
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix
from scipy import linalg
# In[2]:
# vector 생성
myVector = np.array([1,2,3,4])
# matrix 생성
myMatrix = np.matrix(np.random.random((5,5)))
print("vector = \n", myVector)
print("matrix = \n", myMatrix)

# In[ ]:
# 여러 matrix 생성
B = np.array([[1,2,3,4], [5,0,7,8], [9,10,11,12]]) # 2차원으로 생성
eye = np.eye(4)
# Compressed Sparse Row matrix
# 0이 아닌 원소만 저장됩니다.
# 열 우선순서로
sp_matrix = sp.sparse.csr_matrix(B)  # NumPy 배열을 CSR 포맷의 SciPy 희소 행렬로 변환.
print("Scipy의 CSR 행렬:\n{}".format(sp_matrix))  
# In[ ]: 
# Compressed Sparse Column matrix
# 0이 아닌 원소만 저장됩니다.
# 행 우선순서로
sp_matrix = sp.sparse.csc_matrix(B)
print("Scipy의 CSC 행렬:\n{}".format(sp_matrix))
# In[ ]:
 # Dictionary Of Keys matrix  
sp_matrix = sp.sparse.dok_matrix(B)
print("Scipy의 DOK 행렬:\n{}".format(sp_matrix))    # Dictionary Of Keys matrix

# In[ ]:
# vector의 각종 연산
vector1 = np.array([1,2,3])
vector2 = np.array([8,7,6])

# 벡터의 덧셈
vector3 = vector1 + vector2
print(vector3)
# 벡터의 뺄셈
vector4 = vector2 - vector1
print(vector4)
# 벡터곱
dotProduct = np.dot(vector1, vector2)
print(dotProduct)
# 벡터의 cross곱
# Cross를 하기 위해서는 차원의 값이 2 또는 3이어야 한다.
crossProduct = np.cross(vector1, vector2)
print(crossProduct)

# In[ ]:
vector1 = np.array([1,2,3,4])
vector2 = np.array([8,7,6,1])
# Vector dot product = 스칼라곱  A·B
vectorDotProduct = np.vdot(vector1, vector2)
print(vectorDotProduct)
# Inner product = 내적
innerProduct = np.inner(vector1, vector2)
print(innerProduct)
# Outer product = 외적
outerProduct = np.outer(vector1, vector2)
print(outerProduct)

matrix1 = np.matrix(np.eye(3))
matrix2 = np.matrix(np.eye(3))
# Tensor dot product
tensorDotProduct = np.tensordot(matrix1, matrix2)
print(tensorDotProduct)
# Kronecker product = A ⊗ B
kronProduct = np.kron(matrix1, matrix2)
print(kronProduct)

# In[ ]:
matrix1 = np.matrix(np.eye(3))
matrix1[0] = 2
matrix1[1,2] = 8
print(matrix1)
print("----------------------------------\n", end="\n")
# In[ ]:
# Transposition
print(matrix1)
print(matrix1.T)
print("----------------------------------\n", end="\n")
# In[ ]:
# Conjugate transposition
print(matrix1)
print(matrix1.H)
print("----------------------------------\n", end="\n")
# In[ ]:
# Inverse
print(matrix1)
print(matrix1.I)
print("----------------------------------\n", end="\n")
# In[ ]:
# Array
print(matrix1)
print(matrix1.A)


#scipy.linalg.funm(matrix1, lambda x: x+1)