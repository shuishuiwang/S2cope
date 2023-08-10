# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: cal_index.py
 @time: 2023/4/19 15:36
 @productname: langfeng
 @说明：
'''
import matplotlib

import rasterio
from netCDF4 import Dataset
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy
from rasterio.plot import  reshape_as_image
from osgeo import gdal
import os
from skimage import morphology
from skimage.filters import median
from skimage.morphology import disk,ball
import sys
from matplotlib.colors import LinearSegmentedColormap,ListedColormap

sys.path.append("K:\个人代码工具\Module")

hasallndex = ['rgb', 'ndwi', 'fai', 'avw', 'hueangle', 'ndvi', 'afai', 'sipf', 'vb_fah']

np.seterr(divide='ignore',invalid='ignore')
medianyn=True #!进行中值滤波
addrgbdata=True #!添加RGB
matplotlib.use('agg')#!设置不出现figure后端
# import re
# os.environ['PROJ_LIB'] = r'E:\Users\shui\anaconda3\pkgs\proj-9.1.1-heca977f_2\Library\share\proj'
# os.environ['GDAL_DATA'] = r'E:\Users\shui\anaconda3\pkgs\proj-9.1.1-heca977f_2\Library\share'

#!!K:\个人代码工具\Module\cal_RSindex.py
watertypes='lake'#!宽度小于100  ['lake','midianriver','minriver']
clear_glint=True #!是否去除耀斑
Remotesensing='S2'#!['S2','planet']

#!!K:\个人代码工具\Module\outputJPGmodule.py  #!机器学习用于TN，TP
machinelearning=True
norm_dengfen=True
sys.path.append('K:\Project_engpath\WaterqualityDL')
outmodeldir=__import__('S2Config').outmodeldir
parraset=__import__('S2Config').parraset
# from S2Config import *

#!机器学习参数位置
directory={
    # Code存放代码，JPG存在整体大图，Product存在产品及单图
    'Code':{
    },
    'JPG':{
        "Index":{},
        'WQ':{},
    },
    'Product':{
        'Chla':{},
        'SPM':{},
        'vb_fah':{},
        'TN':{},
        'TP':{},
        'NH3':{},
        
    },
}

def cmapcal(cmapa,X=2):
    chlacmaps={}
    for ky in cmapa.keys():
        chlacmaps[ky/X]=cmapa[ky]
    return chlacmaps
def cmapcaljiaxian(cmapa,X=2):
    chlacmaps={}
    for ky in cmapa.keys():
        chlacmaps[ky+X]=cmapa[ky]
    return chlacmaps
def getnorm_cmap(product,means=None):
    if product=='Chla':
        if means<10:
            chlacmaps=cmapcal(chlacmap,X=2)
        elif means<25:
            chlacmaps=cmapcal(chlacmap,X=2)
        else:
            chlacmaps=chlacmap
        if means>80:
            chlacmaps=cmapcal(chlacmap,X=0.5)
        if means>160:
            chlacmaps=cmapcal(chlacmap,X=0.2)
        cmap,norm= getcamp(chlacmaps)
    if product=='SPM':
        if means<10:
            spmcmaps=cmapcal(spmcmap,X=3)
        elif means<25:
            spmcmaps=cmapcal(spmcmap,X=2)
        else:
            spmcmaps=spmcmap
        if means>80:
            spmcmaps=cmapcal(spmcmap,X=0.5)
        if means>160:
            spmcmaps=cmapcal(spmcmap,X=0.2)
        # if means<25:
        # spmcmaps=cmapcal(spmcmap,X=1/3)
        cmap,norm= getcamp(spmcmaps)

    if product=='TN':
        cmap,norm= getcamp(TNcmap)
    if product=='TP':
        cmap,norm= getcamp(TPcmap)
    if product=='NH3N':
        # NH3Ncmaps=cmapcal(NH3Ncmap,X=0.5)

        cmap,norm= getcamp(NH3Ncmap)
    if product=='ndvi':
        cmap,norm= getcamp(ndvicmap)
    if product=='KMNO':
        # KMNOcmaps=cmapcal(KMNOcmap,X=0.5)
        cmap,norm= getcamp(KMNOcmap)
    if product=='Trans':
        # KMNOcmaps=cmapcal(KMNOcmap,X=0.5)
        cmap,norm= getcamp(Transcmap)
    if product=='NTU':
        spmcmaps=cmapcal(spmcmap,X=2/3)
        cmap,norm= getcamp(spmcmaps)
    if product=='DO':
        # spmcmaps=cmapcal(DOcmap,X=2/3)
        cmap,norm= getcamp(DOcmap)
        norm=None
    if norm_dengfen:
        cmap,norm= getcamp(norm_dengfencmap)
        norm=None
    if product=='vb_fah':
        if means<-0.005:
            vb_fahcmaps=cmapcaljiaxian(vb_fahcmap,X=-0.002)
        else:
            vb_fahcmaps=vb_fahcmap
        cmap,norm= getcamp(vb_fahcmaps)        
    return cmap,norm
def getnorm(product,onedata):
    onedata=np.where(np.isinf(onedata),np.nan,onedata)
    
    if product=='SPM' or product=='Chla':
        if np.nanmean(onedata)>75:
            norm=plt.Normalize(0,150)
        elif np.nanmean(onedata)>45:
            norm=plt.Normalize(0,100)
        elif np.nanmean(onedata)>20:
            norm=plt.Normalize(0,50)
        else:
            norm=plt.Normalize(0,25)
    return norm
def Create_dir(dirs):
    for key,da in directory.items():
        dirc = os.path.join(dirs, key)
        if not os.path.exists(dirc):
            os.makedirs(dirc)
        if len(da)!=0:
            for key, da in da.items():
                dirc1 = os.path.join(dirc, key)
                if not os.path.exists(dirc1):
                    os.makedirs(dirc1)
Caldata={
    'Chla':{
        '1':{'exp':'0.025*x**2-0.518*x+3.871','x':'842*(783-705)','source':'喀斯特高原','sensor':'S2','danwei':'ug/L'},
        '2':{'exp':'0.321*x+6.9784','x':'783','source':'喀斯特高原','sensor':'S2','danwei':'ug/L'},
        '3':{'exp':'0.8145*x+2.57','x':'783*842','source':'喀斯特高原','sensor':'S2','danwei':'ug/L'},
        '4':{'exp':'2.718**(-4.7624*x**2+12.798*x-4.6898)','x':'1/(665*705)','source':'GWL硕士论文','sensor':'S2','danwei':'ug/L'},
        '5':{'exp':'2.718**(-5.6449*x**2+14.173*x-5.2135)','x':'1/(665*705)','source':'GWL硕士论文','sensor':'S2','danwei':'ug/L'},
        '6':{'exp':'2.718**(1.102+6.053*x-17.264*x**2+12.647*x**3-2.799*x**4)','x':'490/560','source':'秦皇岛海域','sensor':'S2','danwei':'ug/L'},
        '7':{'exp':'1903*x-1.248','x':'1/490','source':'香港近海海域','sensor':'S2','danwei':'ug/L'},
        '8':{'exp':'1496.352*x**2-2928.304*x+1438.999','x':'705/665','source':'平寨水库','sensor':'S2','danwei':'ug/L'},
        '9':{'exp':'781.251*x**2+75.657*x+8.014','x':'(1/665-1/705)*842','source':'平寨水库','sensor':'S2','danwei':'ug/L'},
        '10':{'exp':'18.87*x**2 + 79.30*x-65.66 ','x':'705/665','source':'','sensor':'S2','danwei':'ug/L'},
        '11':{'exp':'706.55*x**2-264.95*x+ 24.83 ','x':'(705-665)/(705+665)','source':'','sensor':'S2','danwei':'ug/L'},
        '12':{'exp':'172.57*x+ 26.85 ','x':'(1/665-1/705)*740','source':'','sensor':'S2','danwei':'ug/L'},
        # '13':{'exp':'172.57*x+ 26.85 ','x':'(1/665-1/705)*740','source':'','sensor':'S2','danwei':'ug/L'},

    },
    'SPM':{
        '1': {'exp': '955.63*x-1.13', 'x': '560*705', 'source': '珠海市海域悬浮物遥感', 'sensor': 'S2',
              'danwei': 'mg/L'},
        '2': {'exp': '196.27*x-655.13', 'x': '705', 'source': '珠海市海域悬浮物遥感', 'sensor': 'S2',
              'danwei': 'mg/L'},
        '3': {'exp': '1.712*2.71828**(0.003*x)', 'x': '705', 'source': 'GF-1与GF-6WFV', 'sensor': 'S2',
              'danwei': 'mg/L'},
        '4': {'exp': '1.958*2.718**(0.0032*x)', 'x': '710', 'source': 'GF-1与GF-6WFV', 'sensor': 'S2',
              'danwei': 'mg/L'},
        '5': {'exp': '2.718**(4.337*2.718**(-2.416*x))', 'x': '(560-665)/(560+665)', 'source': 'GWL硕士论文', 'sensor': 'S2',
              'danwei': 'mg/L'},
        '6': {'exp': '2.718**(4.339*2.718**(-2.396*x))', 'x': '(560-665)/(560+665)', 'source': 'GWL硕士论文', 'sensor': 'S2',
              'danwei': 'mg/L'},
        
    },
    'Cdom':{},
    
}
[ 'SPM','Chla','TP', 'TN']+['NTU','KMNO','NH3N','Trans']
dataplot={
'Chla':{'dw':'Chla(ug/L)','norm':plt.Normalize(0, 50),'cmap':'jet'          },
'SPM':{'dw':'SPM(mg/L)','norm':plt.Normalize(0, 50),'cmap':'coolwarm'          },
'Cdom':{               },
'vb_fah':{'dw':'水华指数','norm':None,'cmap':'jet' },
'TP':{'dw':'总磷(mg/L)','norm':None,'cmap':'coolwarm' },
'TN':{'dw':'总氮(mg/L)','norm':None,'cmap':'coolwarm' },
'ndvi':{'dw':'归一化植被指数','norm':None,'cmap':'coolwarm' },
'NTU':{'dw':'浊度(NTU)','norm':None,'cmap':'coolwarm' },
'KMNO':{'dw':'高锰酸盐指数(mg/L)','norm':None,'cmap':'coolwarm' },
'NH3N':{'dw':'氨氮(mg/L)','norm':None,'cmap':'coolwarm' },
'Trans':{'dw':'透明度(cm)','norm':None,'cmap':'coolwarm' },
'DO':{'dw':'溶解氧(mg/L)','norm':None,'cmap':'coolwarm' },
}
def getcamp(vb_fahcamp):
    """vb_fahcamp={
  -0.006: (117, 57, 190, 255),
  -0.005: (46, 26, 225, 255),
  -0.004: (26, 130, 215, 255),
  -0.003: (19, 208, 255, 255),
  -0.002: (21, 255, 142, 255),
  -0.001: (219, 255, 36, 255),
  0: (255, 202, 132, 255),
  0.005: (255, 112, 56, 255),
  0.01: (255, 32, 3, 255)
}

    Args:
        vb_fahcamp (_type_): _description_

    Returns:
        _type_: _description_
    """
    poss=np.array(list(vb_fahcamp.keys()))
    norm=plt.Normalize(poss.min(axis=0),poss.max(axis=0))
    print(poss.max(axis=0))
    X_std = (poss - poss.min(axis=0)) / (poss.max(axis=0) - poss.min(axis=0))
    colors = []
    for r,pos  in enumerate(list(vb_fahcamp.keys())):
        rgba=vb_fahcamp[pos]
        rgba=[s/255 for s in rgba]
        colors.append((X_std[r],rgba))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return cmap,norm
vb_fahcmap={
  -0.006: (117, 57, 190, 255),
  -0.005: (46, 26, 225, 255),
  -0.004: (26, 130, 215, 255),
  -0.003: (19, 208, 255, 255),
  -0.002: (21, 255, 142, 255),
  -0.001: (219, 255, 36, 255),
  0: (255, 202, 132, 255),
  0.005: (255, 112, 56, 255),
  0.01: (255, 32, 3, 255)
}
ndvicmap={
  -0.6: (117, 57, 190, 255),
  -0.5: (46, 26, 225, 255),
  -0.4: (26, 130, 215, 255),
  -0.3: (19, 208, 255, 255),
  -0.2: (21, 255, 142, 255),
  -0.1: (219, 255, 36, 255),
  0: (255, 202, 132, 255),
  0.3: (255, 112, 56, 255),
  0.5: (255, 32, 3, 255)
}
Transcmap={
  0: (117, 57, 190, 255),
  5: (46, 26, 225, 255),
  10: (26, 130, 215, 255),
  15: (19, 208, 255, 255),
  20: (21, 255, 142, 255),
  30: (219, 255, 36, 255),
  50: (255, 202, 132, 255),
  70: (255, 112, 56, 255),
  100: (255, 32, 3, 255)
}


spmcmap={
  0: (117, 57, 190, 255),
  5: (46, 26, 225, 255),
  10: (26, 130, 215, 255),
  15: (19, 208, 255, 255),
  20: (21, 255, 142, 255),
  30: (219, 255, 36, 255),
  50: (255, 202, 132, 255),
  70: (255, 112, 56, 255),
  100: (255, 32, 3, 255)
}
nh3ncmap={
  0: (117, 57, 190, 255),
  5: (46, 26, 225, 255),
  10: (26, 130, 215, 255),
  15: (19, 208, 255, 255),
  20: (21, 255, 142, 255),
  30: (219, 255, 36, 255),
  50: (255, 202, 132, 255),
  70: (255, 112, 56, 255),
  100: (255, 32, 3, 255)
}
chlacmap={
  0: (117, 57, 190, 255),
  5: (46, 26, 225, 255),
  10: (26, 130, 215, 255),
  15: (19, 208, 255, 255),
  20: (21, 255, 142, 255),
  30: (219, 255, 36, 255),
  50: (255, 202, 132, 255),
  70: (255, 112, 56, 255),
  100: (255, 32, 3, 255)
}
TPcmap={
  0: (117, 57, 190, 255),
  0.01: (46, 26, 225, 255),
  0.03: (26, 130, 215, 255),
  0.05: (19, 208, 255, 255),
  0.07: (21, 255, 142, 255),
  0.09: (219, 255, 36, 255),
  0.15: (255, 202, 132, 255),
  0.25: (255, 112, 56, 255),
  0.4: (255, 32, 3, 255)
}
norm_dengfencmap={
  0: (117, 57, 190, 255),
  0.1: (46, 26, 225, 255),
  0.2: (26, 130, 215, 255),
  0.3: (19, 208, 255, 255),
  0.4: (21, 255, 142, 255),
  0.5: (219, 255, 36, 255),
  0.6: (255, 202, 132, 255),
  0.7: (255, 112, 56, 255),
  0.73: (255, 32, 3, 255)
}
NH3Ncmap={
  0: (117, 57, 190, 255),
  0.05: (46, 26, 225, 255),
  0.1: (26, 130, 215, 255),
  0.2: (19, 208, 255, 255),
  0.3: (21, 255, 142, 255),
  0.5: (219, 255, 36, 255),
  0.9: (255, 202, 132, 255),
  1.5: (255, 112, 56, 255),
  3: (255, 32, 3, 255)
}
TNcmap={
  0: (117, 57, 190, 255),
  0.2: (46, 26, 225, 255),
  0.4: (26, 130, 215, 255),
  0.6: (19, 208, 255, 255),
  0.8: (21, 255, 142, 255),
  1: (219, 255, 36, 255),
  2: (255, 202, 132, 255),
  4: (255, 112, 56, 255),
  5: (255, 32, 3, 255)
}
KMNOcmap={
  0: (117, 57, 190, 255),
  0.5: (46, 26, 225, 255),
  1: (26, 130, 215, 255),
  2: (19, 208, 255, 255),
  3: (21, 255, 142, 255),
  5: (219, 255, 36, 255),
  6: (255, 202, 132, 255),
  7: (255, 112, 56, 255),
  10: (255, 32, 3, 255)
}
DOcmap={
  0: (117, 57, 190, 255),
  0.5: (46, 26, 225, 255),
  1: (26, 130, 215, 255),
  2: (19, 208, 255, 255),
  3: (21, 255, 142, 255),
  5: (219, 255, 36, 255),
  6: (255, 202, 132, 255),
  7: (255, 112, 56, 255),
  10: (255, 32, 3, 255)
}

