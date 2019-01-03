#!/usr/bin/env python
# coding: utf-8

# In[1]:
# numpy
import numpy as np
# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# scipy
import scipy as sp
import scipy.stats
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import dok_matrix
from scipy import linalg
from scipy.io import netcdf
# netcdf
import netCDF4
from netCDF4 import Dataset
# pandas
import pandas as pd
# else
import timeit
import datetime as dt

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

# In[3]:
# netcdf 내부 정보 획득하는 함수 선언
'''
ncInfo는 불러온 nc파일의 dimensions(차원), variables(변수), attribute information(속성 정보)를 리턴한다.
@param
nc_obj : netcdf파일의 Dataset
verb : Boolean 값. 각 속성에 대한 출력 여부

@return
nc_attrs : list(global attributes)
nc_dims : list(dimensions)
nc_vars : list(variables)
'''
def ncInfo(nc_obj, verb=True):
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key
        @param
        key(unicode) : netcdf파일의 Dataset의 variables들의 key 
        """
        try:
            print("\t\ttype:", repr(nc_obj.variables[key].dtype))
            for ncattr in nc_obj.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_obj.variables[key].getncattr(ncattr)))
        except KeyError:    # 입력받은 키가 없는 경우
            print("WARNING: %s 해당 key인 attribute가 존재하지 않습니다." % key)

    # NetCDF global attributes
    nc_attrs = nc_obj.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_obj.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_obj.dimensions]  # dimensions
    # Dimension shape information
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_obj.dimensions[dim]))
            print_ncattr(dim)
    # Variable information
    nc_vars = [var for var in nc_obj.variables]  # variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_obj.variables[var].dimensions)
                print("\t\tsize:", nc_obj.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

# In[ ]:
# netcdf 파일 불러오기
dataset = Dataset('test2.nc')
print(dataset.file_format) 
# In[ ]:
# 불러온 파일 차원 파악하기
print(dataset.dimensions.keys())    # 각 차원의 이름 파악
print(dataset.dimensions['lon'])   # 차원의 이름을 가지고 크기 파악
# In[ ]:
# 불러온 파일 variables 파악하기
print(dataset.variables.keys()) # 전체 variables 이름
print(dataset.variables['lon']) # variables의 특정 이름에 대한 정보
# In[ ]:
# Global attributes 파악하기
# 불러온 파일의 모든 global attributes 찾기
for attr in dataset.ncattrs():
   print(attr, " = ", getattr(dataset, attr))
# In[ ]:
# 특정 attribute의 conventions
print(dataset.description)  # conventions attribute
# In[ ]:
# 위에서 정의한 ncInfo 함수를 실행
# 각각의 정보를 list로 담으면서 각각의 내역 출력
nc_attrs, nc_dims, nc_vars = ncInfo(dataset)
# In[ ]:
# 각 variables의 data를 array로 가져오기
lats = dataset.variables['lat'][:] 
lons = dataset.variables['lon'][:]
time = dataset.variables['time'][:]
air = dataset.variables['air'][:]  # shape is time, lat, lon as shown above
print("lat variables 내 존재하는 값들")
print(lats)
print("---------------------------------------------------------------------------------")
print("lons variables 내 존재하는 값들")
print(lons)
print("---------------------------------------------------------------------------------")
print("time variables 내 존재하는 값들")
print(time)
print("---------------------------------------------------------------------------------")
print("air variables 내 존재하는 값들")
print(air)
# In[ ]:
# nc file의 'time' 분석
time_idx = 120                  # 저장된 time에 따라 정해진 기준 날로부터 일 수
offset = dt.timedelta(hours=48) # hours(시간)에 따라 지난 날짜와 시간 값을 알려줌
# nc file에 존재하는 모든 time 값들을 전부 년월일 데이터로 변환
dt_time = [dt.date(1, 1, 1) + dt.timedelta(hours=t) - offset\
           for t in time]
cur_time = dt_time[time_idx] # time_idx에 따른 지정 년도의 년월일 데이터
print(offset)
print("-------------------------------------------")
for val in dt_time:
    print(val, "\n")
print("-------------------------------------------")
print(cur_time)
# In[ ]:
# In[ ]:
# matplotlib으로 위에 지정한 날짜의 전세계 온도에 대한 plot 진행
fig = plt.figure()
fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
# Setup the map. See http://matplotlib.org/basemap/users/mapsetup.html
# for other projections.
m = Basemap(projection='moll', llcrnrlat=-90, urcrnrlat=90,\
            llcrnrlon=0, urcrnrlon=360, resolution='c', lon_0=0)
m.drawcoastlines()
m.drawmapboundary()
# Make the plot continuous
air_cyclic, lons_cyclic = addcyclic(air[time_idx, :, :], lons)
# Shift the grid so lons go from -180 to 180 instead of 0 to 360.
air_cyclic, lons_cyclic = shiftgrid(180., air_cyclic, lons_cyclic, start=False)
# Create 2D lat/lon arrays for Basemap
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
# Transforms lat/lon into plotting coordinates for projection
x, y = m(lon2d, lat2d)
# Plot of air temperature with 11 contour intervals
cs = m.contourf(x, y, air_cyclic, 11, cmap=plt.cm.Spectral_r)
cbar = plt.colorbar(cs, orientation='horizontal', shrink=0.5)
cbar.set_label("%s (%s)" % (nc_fid.variables['air'].var_desc,\
                            nc_fid.variables['air'].units))
plt.title("%s on %s" % (nc_fid.variables['air'].var_desc, cur_time))