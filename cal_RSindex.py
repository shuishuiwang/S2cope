# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: cal_index.py
 @time: 2023/4/19 15:36
 @productname: langfeng
 @说明：
'''
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
from skimage.morphology import disk
from Config import *


def gethueangle_Xan(XAn, wavelengthd):
    """
    arr:arr.shape=(620, 642, 218)
    data:data=pd.read_csv('H:\无人机数据处理过程\图像处理\颜色匹配函数.csv')
    wavelengthd：影像每个波段对应的波长
    """
    data = pd.read_csv('K:\工作\蓝丰\洪泽湖\Code\颜色匹配函数.csv', encoding='utf_8_sig')
    N, B = XAn.shape
    from numpy import arange, array, exp
    def extrap1d(interpolator):
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
            elif x > xs[-1]:
                return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            else:
                return interpolator(x)

        def ufunclike(xs):
            return array(list(map(pointwise, array(xs))))

        return ufunclike
    # view_cube(image)
    from scipy import interpolate as intp
    wavelengthd = np.array(wavelengthd)
    wavelength = np.array(data['wavelength'])
    xm = np.array(data['x|'])
    ym = np.array(data['y|'])
    zm = np.array(data['z|'])
    fx = intp.interp1d(wavelength, xm, kind='slinear')
    f_x = extrap1d(fx)
    fxm = f_x(wavelengthd[0:B])
    fy = intp.interp1d(wavelength, ym, kind='slinear')
    f_y = extrap1d(fy)
    fym = f_y(wavelengthd[0:B])
    fz = intp.interp1d(wavelength, zm, kind='slinear')
    f_z = extrap1d(fz)
    fzm = f_z(wavelengthd[0:B])
    # plt.plot(wavelengthd[0:B],fxm, wavelengthd[0:B],fym,wavelengthd[0:B],fzm  )
    # plt.show()
    # fz(wavelengthd)
    k = 1.729023316
    jieguo = []
    # X=np.sum(fxm*image,2)
    fxm = np.reshape(fxm, [1, B])
    fym = np.reshape(fym, [1, B])
    fzm = np.reshape(fzm, [1, B])
    X0 = fxm * XAn * k
    Xsum = np.sum(X0, 1)
    ysum = np.sum(fym * XAn * k, 1)
    Zsum = np.sum(fzm * XAn * k, 1)
    x = Xsum / (Xsum + ysum + Zsum)
    y = ysum / (Xsum + ysum + Zsum)
    z = Zsum / (Xsum + ysum + Zsum)
    a = 180 + np.arctan2(x - 0.3333, y - 0.3333) / 3.1415926 * 180
    # print(np.min(a))
    # print(np.max(a))
    index=['hueangle']
    return a


class calindex:
    # addrgbdata=True#!是否叠加真彩色影像
    def __init__(self,data,wavelength):
        self.data=data
        self.wavelength=np.array(wavelength)
    def bywave(self,wave):
        index=np.argmin(np.abs(self.wavelength-wave))
        # print(index)
        outdata=self.data[index]
        return outdata
    def show(self,showdata,per):
        s1, s2 = np.nanpercentile(showdata, per), np.nanpercentile(showdata, 100-per)
        da = np.clip(showdata, s1, s2)
        return  da
    def view(self,out,nrom=None):
        import matplotlib
        matplotlib.use("Tkagg")
        plt.figure()
        plt.imshow(out,'jet',norm=nrom)
        plt.colorbar()
        plt.show()
    def QAforS2(self):
        if Remotesensing=='S2':
            ndwi=self.ndwi()
            # self.view(self.getrgbdata())
            # self.view(ndwi)
            # ndbi=self.ndbi()
            percent=np.nansum(self.bywave(560)*ndwi < 300 / (10000 * np.pi))/np.nansum(ndwi)#!水中阴影占比，可能对于极度清净水体有害
            # print('shadepercent:',percent)
            if percent>0.1:#!如果阴影大于0.1，则说明这种水体无阴影影像，光谱为阴影
                # print('')
                cloud_build =(self.bywave(945) > 1000 / (10000 * np.pi))
                fcloud= 1-cloud_build
                QA = fcloud
                QA = np.where(QA > 0, 1, np.nan)
            else:
                shade = 1 - (self.bywave(560) < 300 / (10000 * np.pi))
                cloud_build =(self.bywave(945) > 1200 / (10000 * np.pi))
                fcloud= 1-cloud_build
                QA = fcloud * shade
                QA = np.where(QA > 0, 1, np.nan)
            if self.iszaohua():#!表明存在藻华 
                # print('可能存在藻华')
                shade = 1 - (self.bywave(560) < 300 / (10000 * np.pi))
                cloud_build =(self.bywave(1610) > 1500 / (10000 * np.pi))
                fcloud= 1-cloud_build
                QA = fcloud * shade
                QA = np.where(QA > 0, 1, np.nan)
            if True in np.unique(self.wavelength>1500):
                if clear_glint==True:
                    if watertypes=='lake':
                        glint_thresh=900
                    else:
                        glint_thresh=700
                    goodwater=(self.bywave(1610) < glint_thresh / (10000 * np.pi))#!B12小于700
                    goodwater=np.where(goodwater==0,np.nan,goodwater)
                    QA=QA*goodwater
            return QA
        elif Remotesensing=='planet':
            cloud=self.bywave(866)>3000/ (10000 * np.pi)#!需要修改
            mask=np.isnan(self.bywave(866))
            QA=(1-cloud)*(1-mask)
            QA=np.where(QA>0,1,np.nan)
            return QA
    def getall_index(self, need=None,date=None ):

        # count = len(ndwi[np.isnan(ndwi)])
        allndex = {
            'rgb': {'ds': reshape_as_image(self.rgb(3)), 'norm': None,'zwname':'真实影像'},
            'ndwi': {'ds':  self.ndwi, 'norm': None,'zwname':'归一化水体指数'},
            'fai': {'ds': self.fai, 'norm': None,'zwname':'水华指数'},
            'avw': {'ds': self.avw, 'norm': None,'zwname':'表观波长'},
            'hueangle': {'ds': self.hueangle, 'norm': None,'zwname':'色相角'},
            'ndvi': {'ds': self.ndvi, 'norm': plt.Normalize(-0.4,0.4),'zwname':'归一化植被指数'},
            'afai': {'ds': self.afai, 'norm': plt.Normalize(0, 0.04),'zwname':'水华指数'},
            'sipf': {'ds': self.sipf, 'norm': None,'zwname':'水华指数'},
            'vb_fah': {'ds': self.vb_fah, 'norm': plt.Normalize(-0.006, 0.0),'zwname':'水华指数'},
            'CI': {'ds': self.CI, 'norm':None,'zwname':'蓝藻指数'},
        }
        if need is None:
            lens=len(allndex.keys())
            cols=int(np.sqrt(lens))
            outdata={}
            fig, axs = plt.subplots(cols, round(np.ceil(lens/cols)))
            quchu=-(cols*round(np.ceil(lens/cols))-lens)
            # axs.ravel()[quchu:]=None
            if lens>1:
                for ax in axs.ravel()[quchu:]:
                    ax.axis('off')
            for r,name in enumerate(allndex.keys()):
                norm=allndex[name]['norm']
                zwname=allndex[name]['zwname']
                norm=None
                try:
                    cmap,norm=getnorm_cmap(name)
                except:
                    norm = allndex[name]['norm']
                    cmap='coolwarm'
                    pass
                ax=axs.ravel()[r]
                if name == 'rgb':
                    data=self.getrgbdata(savepath=None)                      
                    ax.imshow(data)
                else:
                    # QAforS2 = self.QAforS2()
                    # ndwi = self.ndwi()
                    data = np.round(allndex[name]['ds'].__call__(), 3)
                    data=self.show(data*self.ndwi()*self.QAforS2(),2)
                    # print(np.nanmean(data))
                    if name=='ndvi':
                        # norm=norm
                        vmin=None
                        # print('NDVI',np.nanmax(data))
                    else:
                        # norm=None
                        vmin=None
                    # if self.addrgbdata==True:
                    #     ax.imshow(self.getrgbdata(savepath=None))    
                    nx=ax.imshow(data,cmap,norm)
                    fig.colorbar(mappable=nx,ax=ax,shrink=0.68)
                    ax.axis(False)
                ax.set_title(name,family='Simsun')
                if date is not None:
                    ax.set_title(date)
                outdata[name]=data
            plt.tight_layout()
            if date is not None:
                plt.title(date)
            # plt.show()
            return  fig,outdata
        
        else:
            lens = len(need)
            cols = int(np.sqrt(lens))
            outdata = {}
            fig, axs = plt.subplots(cols,  round(np.ceil(lens/cols)))
            quchu=-(cols*round(np.ceil(lens/cols))-lens)
            if lens>1:
                for ax in axs.ravel()[quchu:]:
                    ax.axis('off')
            # axs.ravel()[quchu:]=None
            for r, name in enumerate(need):
                zwname=allndex[name]['zwname']

                # norm = None
                try:
                    ax = axs.ravel()[r]
                except :
                    ax = axs
                if name == 'rgb':
                    data=self.getrgbdata(savepath=None)                    
                    ax.imshow(data)
                else:
                    data = np.round(allndex[name]['ds'].__call__() , 3)
                    data = self.show(data *self.ndwi()*self.QAforS2(), 2)
                    # print(np.nanmean(data))
                    if name == 'ndvi':
                        # norm=None
                        vmin = None
                        
                        # print('NDVI', np.nanmax(data))
                    else:
                        # norm=None
                        vmin = None
                    if addrgbdata==True:
                        ax.imshow(self.getrgbdata(savepath=None))  
                    if medianyn:
                        data=self.show(data,2)
                        data = median(data, disk(5))
                        # matplotlib.use('Tkagg')
                    try:
                        means=np.nanmean(data)
                        cmap,norm=getnorm_cmap(name,means=means)
                        

                    except:
                        norm = allndex[name]['norm']
                        cmap='coolwarm'
                        pass
                    nx = ax.imshow(data, cmap,norm)
                    
                    # plt.imshow(data,cmap=cmap,norm=norm)
                    # plt.show()
                    
                    cb=fig.colorbar(mappable=nx, ax=ax,shrink=0.68)

                    [s.set_family('times new roman') for s in cb.ax.get_yticklabels()]
                    cb.set_label(zwname,family='Simsun')
                    ax.axis(False)
                # ax.set_title(zwname,family='Simsun')
                if date is not None:
                    ax.set_title(date,family='times new roman')
                outdata[name]=data
            plt.tight_layout()
            # fig.show()
            return fig, outdata
    def calbyexp(self,exp):
        # exp = '842*(783-705)'
        de = {}
        nedwve = re.findall('\d+', exp)

        for r, s in enumerate(nedwve):
            wave=int(s)
            if wave<400:
                # print('wave:',wave)
                continue
            data=self.bywave(wave)
            exp = exp.replace(s, 'B' + str(r))
            new = {'B' + str(r): data}
            de.update(new)
        out = eval(exp, de)

            # out==data[0]*(data[1]-data[2])
        return out
    def calWQbyexp(self,exp,x_exp):
        if isinstance(x_exp,str):
            x=self.calbyexp(x_exp)
            outdata=eval(exp)
            return outdata
        else:
            x=x_exp
            outdata = eval(exp,x)
            return outdata
    def calpro(self,product,need=None):
        outpro = {}
        # product = 'Chla'
        # print(product,Caldata)
        if need is None:
            needkey=Caldata[product].keys()
        else:
            needkey=[]
            for ke in Caldata[product].keys():
                Ser = Caldata[product][ke]
                outname = ke + '_' + Ser['source'] + Ser['x']
                for name in need:
                   if name==outname:
                       if ke  not in needkey:
                           needkey.append(ke)
        for ke in needkey:
            Ser = Caldata[product][ke]
            exp = Ser['exp']
            if exp != '':
                x = Ser['x']
                outname = ke+'_'+Ser['source'] + Ser['x']

                out = self.calWQbyexp(exp, x)
                # needout = self.Cal.show(out, 2)
                # print(outname,np.nanmedian(out))
                outpro[outname] = np.round(out,3)
        return outpro
    def getshowall_WQ(self, outpro=None):
        
        lens=len(outpro.keys())+1
        cols=int(np.sqrt(lens))
        fig, axs = plt.subplots(cols, round(np.ceil(lens/cols)),figsize=(15,10))
        quchu=-(cols*round(np.ceil(lens/cols))-lens)
        # axs.ravel()[quchu:]=None
        if lens>1:
            for ax in axs.ravel()[quchu:]:
                ax.axis('off')
        axs.ravel()[0].imshow(self.getrgbdata())
        # axs.ravel()[0].title()
        QAall=self.ndwi()*self.QAforS2()
        
        for r, name in enumerate(outpro.keys()):
            data = outpro[name]
            ax = axs.ravel()[r+1]
            data = self.show(data*QAall , 3)
            # print(np.nanmean(data))
            # print(name,np.nanmean(data))
            nx = ax.imshow(data, 'jet')
            cbar = fig.colorbar(mappable=nx, ax=ax,shrink=0.58)
            cbar.set_label(name, family='Simsun', loc='center',fontsize=10)
            ax.axis(False)

        # plt.tight_layout()
        # fig.show()
        # import matplotlib
        # matplotlib.use('Tkagg')
        return fig
    def mosaic_chla(self,QA=True):
        
        need=['9_平寨水库(1/665-1/705)*842','10_705/665','13_(1/665-1/705)*740']
        data = self.calpro('Chla', need)
        if QA:
            QAall=self.ndwi()*self.QAforS2()
            data=QAall*np.nanmin(np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()] ),axis=0)
        else:
            data=np.nanmin(np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()] ),axis=0)
        data=np.where(np.isinf(data),np.nan,data)
        data=self.show(data,1)
        # if np.nanmean(data)>25:
        #     data=QAall*np.nanmin(np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()] ),axis=0)/1.5
        if np.nanpercentile(data,2)<0:
            need=['9_平寨水库(1/665-1/705)*842','12_(1/665-1/705)*740']
            data = self.calpro('Chla', need)
            # QAall=self.ndwi()*self.QAforS2()
            if QA:
                data=QAall*np.nanmin(np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()] ),axis=0)
            else:
                data=np.nanmin(np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()] ),axis=0)
            data=np.where(np.isinf(data),np.nan,data)
            data=np.where(data<0,0,data)
            data=self.show(data,1)

        # matplotlib.use('Tkagg')
        # cp,norm=getnorm_cmap('Chla',np.nanmean(data))
        # plt.imshow(data,cp,norm)
        # plt.colorbar()
        # plt.show()
        
        data=np.expand_dims(data,axis=0)
        return data     
    def sertmodel(self,QA=True):
        # 665 0.0712 29.5423
        # 740 0.0908 4.7917
        # 783 0.0971 4.4241
        # print('yunxin')
        if QA:       
            QAdata=self.ndwi()*self.QAforS2()
        p=[1.57,1.61,1.92,2.08]
        q=[99.87 ,53.40 ,8.86,7.48]
        #!第一个比值是悬浮泥沙浓度，第二个比值及后续的均为叶绿素相关，这样总的是悬浮物浓度
        bandList=[665,705,740,783]
        row,col=self.bywave(665).shape
        allBandSPM=np.zeros((len(bandList),row,col))
        for i in range(0,4):
            p0 =p[i]
            q0=q[i]
            rrs = self.bywave(bandList[i])
            rrs = np.where(rrs <= 0, np.nan, rrs)
            rrs560=self.bywave(560)
            rrs560=np.where(rrs560 <= 0, np.nan, rrs560)
            x=rrs/rrs560
            if QA:       
                out=self.show(2*p0*x/(q0*(p0-x)**2)*QAdata,2)
            else:
                out=2*p0*x/(q0*(p0-x)**2)
            # print(np.unique(out))
            allBandSPM[i]=out.astype(np.float32)
        allBandSPM=np.where(allBandSPM<=0,np.nan,allBandSPM)
        allBandSPM=np.nanmean(allBandSPM,axis=0)#!使用均值
        outSPMData=np.expand_dims(allBandSPM*1000,axis=0)
        # self.view(SPMData,out)
        return outSPMData
    
    def iszaohua(self):
        if True in np.unique(self.wavelength>1500):
            bs = (self.bywave(550) - self.bywave(1600)) / (self.bywave(550) + self.bywave(1600))
        else:
            bs=(self.bywave(550) - self.bywave(745)) / (self.bywave(550) + self.bywave(745))

        # bs=(Cal.bywave(550) - Cal.bywave(1600)) / (Cal.bywave(550) + Cal.bywave(1600))
        
        ndvi=self.ndvi()*np.where(bs> 0.1, 1, np.nan)
        vb_fah=self.vb_fah()*np.where(bs> 0.0, 1, np.nan)
        if np.nansum(ndvi>0.1)/np.nansum(np.where(bs> 0.3, 1, np.nan))>0.25 or np.nansum(vb_fah>0.01)/np.nansum(np.where(bs> 0.3, 1, np.nan))>0.25:#!表明存在藻华
            # print('存在藻华')
            return True
    def ndwi(self,postclasscope=False):
        if Remotesensing=='S2':
            if True in np.unique(self.wavelength>1500):
                bs = (self.bywave(550) - self.bywave(1600)) / (self.bywave(550) + self.bywave(1600))
                if watertypes=='lake':
                    if self.iszaohua():#!表明存在藻华
                        ndwi = np.where(bs> 0.2, 1, np.nan)
                    else:
                        ndwi = np.where(bs> 0.35, 1, np.nan)
                elif  watertypes=='minriver':
                    ndwi = np.where(bs> -0.1, 1, np.nan)
                    
            else:
                # S2bs=(self.bywave(550) - self.bywave(745)) / (self.bywave(550) + self.bywave(745))#!S2
                bs=(self.bywave(533) - self.bywave(866)) / (self.bywave(533) +self.bywave(866))#!planet
                if self.iszaohua():#!表明存在藻华
                    ndwi = np.where(bs> 0, 1, np.nan)
                else:
                    ndwi = np.where(bs> 0, 1, np.nan)
            if postclasscope==True:
                try:
                    if watertypes=='lake':
                        ndwi = self.postclasscope(ndwi, removeholethresh=50, removesmallobject=1000)
                    elif watertypes=='minriver':
                        ndwi = self.postclasscope(ndwi, removeholethresh=10, removesmallobject=5)
                except:
                    pass
            return ndwi
        elif Remotesensing=='planet':
            bs=(self.bywave(533) - self.bywave(866)) / (self.bywave(533) +self.bywave(866))
            if watertypes=='lake':
                if self.iszaohua():#!表明存在藻华
                    ndwi = np.where(bs> 0.1, 1, np.nan)
                else:
                    ndwi = np.where(bs> 0.15, 1, np.nan)
            elif  watertypes=='minriver':
                ndwi = np.where(bs> -0.15, 1, np.nan)
            if np.nansum(ndwi)<np.prod(bs.shape)*0.02:#!如果ndwi识别为水体的像素少于总数5%，说明存在问题
                print('ndwi存在问题，采用707nm小于1200算法')
                water=self.bywave(707)<1000/ (10000 * np.pi)
                ndwi=np.where(water> 0, 1, np.nan)
            if postclasscope==True:
                try:
                    if watertypes=='lake':
                        ndwi = self.postclasscope(ndwi, removeholethresh=50, removesmallobject=1000)
                    elif watertypes=='minriver':
                        ndwi = self.postclasscope(ndwi, removeholethresh=10, removesmallobject=10)
                except:
                    pass
            return ndwi
    def CI(self):
        #Evaluation of a satellite-based cyanobacteria bloom detection algorithm using field-measured microcystin data
        #OLCI/MERIS
        bs= (self.bywave(665) - self.bywave(620))+ (self.bywave(681) - self.bywave(1600))*(665-620)/(681-620)
        return bs
    def afai(self):
        # 洱海藻华水花时空
        #OLCI/MERIS
        bs = self.bywave(754) - (self.bywave(665)+(754-665)/(865-665)*(self.bywave(865)-self.bywave(665)))
        return bs
    def sipf(self):
        # 洱海藻华水花时空
        bs = self.bywave(665) - (self.bywave(620)+(665-620)/(681-620)*(self.bywave(681)-self.bywave(620)))
        return bs
    def ndbi(self):
        NDBI = (self.bywave(1665)-self.bywave(842)) / (self.bywave(1665)+self.bywave(842))
        return NDBI
    def ndvi(self):
        # if
        bs = (self.bywave(842) - self.bywave(665)) / (self.bywave(842) + self.bywave(665))
        return bs
    def fai(self):
        # if

        bs = (self.bywave(842) - self.bywave(665))+(self.bywave(560)+ self.bywave(1610))*(self.bywave(842) - self.bywave(665)) / (self.bywave(1610) - self.bywave(665))

        return bs
    def vb_fah(self):
        # 基于FAI方法的洱海蓝藻水华遥感监测
        bs=self.bywave(842)-(
            self.bywave(665)+(self.bywave(1610)-self.bywave(665))*(842-665)/(1610-665)
        )

        return bs
    def hueangle(self):
        # data = pd.read_csv('K:\无人机数据处理过程\图像处理\颜色匹配函数.csv', encoding='utf_8_sig')
        index = np.where((self.wavelength > 450) & (self.wavelength < 750))
        # print(index)
        wavelength = self.wavelength[index]
        rrsdata = self.data[index]
        b, c, l = rrsdata.shape
        XAn = reshape_as_image(rrsdata).reshape((-1, b))
        out = gethueangle_Xan(XAn, wavelength)
        outdata = out.reshape((c, l))
        return outdata
    def avw(self, channel='first', reshape=None):
        index=np.where((self.wavelength>400)&(self.wavelength<750))
        wavelength=self.wavelength[index]
        
        rrsdata=self.data[index]
        if channel == 'first':
            if reshape:
                ds = np.reshape(rrsdata, (12, -1))
                avw_numerator = np.sum(ds, axis=0, dtype=float)
                avw_denotemp = np.zeros(ds.shape)
                for i, wl in enumerate(wavelength):
                    avw_denotemp[i, :] = ds[i, :] / float(wl)
                avw_denominator = np.sum(avw_denotemp, axis=0, dtype=float)
                avw_denominator[avw_denominator <= 0] = np.nan
                avw = avw_numerator / avw_denominator
                avw = np.reshape(avw, (58, 52))
            else:

                # rrsdata = np.array(das.asarray())
                avw_numerator = np.sum(rrsdata, axis=0, dtype=float)
                # shapa=(rrsdata.shape)
                avw_denotemp = np.zeros(rrsdata.shape)
                for i, wl in enumerate(wavelength):
                    avw_denotemp[i, :, :] = rrsdata[i, :, :] / float(wl)
                avw_denominator = np.sum(avw_denotemp, axis=0, dtype=float)
                avw_denominator[avw_denominator <= 0] = np.nan
                avw = avw_numerator / avw_denominator
        else:
            # rrsdata = np.array(das.asarray())
            avw_numerator = np.sum(rrsdata, axis=2, dtype=float)
            # shapa=(rrsdata.shape)
            avw_denotemp = np.zeros(rrsdata.shape)
            for i, wl in enumerate(wavelength):
                avw_denotemp[:, :, i] = rrsdata[:, :, i] / float(wl)
            avw_denominator = np.sum(avw_denotemp, axis=2, dtype=float)
            avw_denominator[avw_denominator <= 0] = np.nan
            avw = avw_numerator / avw_denominator
        return avw
    def getrgbdata(self,savepath=None):
        self.cloud=(self.bywave(945) > 4000 / (10000 * np.pi))#!重云                  
        if np.nansum(self.cloud)/np.nansum(~np.isnan(self.ndwi(postclasscope=True)))>0.15:
            fenge=True
            if fenge:
                ndwi=self.ndwi(postclasscope=False)
                waterdatapd = self.data*(~self.cloud)*(~np.isnan(ndwi))#!水体
                waterdata=reshape_as_image(self.rgb(0.1,needata=waterdatapd,savepath=None))
                self.bulid=self.data*(~self.cloud)*(np.isnan(ndwi))#!建筑物或土壤
                buliddata=reshape_as_image(self.rgb(2,needata=self.bulid,savepath=None))
                clouddata=reshape_as_image(self.rgb(2,needata=self.data*self.cloud,savepath=None))
                waterdata[np.isnan(waterdata)]=0
                buliddata[np.isnan(buliddata)]=0
                data=clouddata+waterdata+buliddata
            else:
                clouddata=reshape_as_image(self.rgb(2,needata=self.data*self.cloud,savepath=None))
                self.otherdata=self.data*(~self.cloud)#!建筑物或土壤
                otherdata=reshape_as_image(self.rgb(2,needata=self.otherdata,savepath=None))
                data=clouddata+otherdata
                
            
            
            # pd=(~self.cloud)*(~np.isnan(self.ndwi(postclasscope=False)))+(~self.cloud)*(np.isnan(self.ndwi(postclasscope=False)))+self.cloud
        else:
            data = reshape_as_image(self.rgb(3))
        if savepath is None:
            return data
        else:
            plt.figure()
            ax=plt.imshow(data)
            ax.axes.axis(False)
            plt.tight_layout()
            plt.savefig(savepath,format='jpg',dpi=500,pad_inches=0.05,bbox_inches='tight')
            plt.show()
    def rgb(self,per,needata=None,savepath=None):
        index1 = np.argmin(np.abs(np.array(self.wavelength) - 490))
        index2 = np.argmin(np.abs(np.array(self.wavelength) - 555))
        index3 = np.argmin(np.abs(np.array(self.wavelength) - 660))
        index = [index3, index2, index1]
        # print(index)
        if needata is None:
            needata = self.data[index]
        else:
            needata= needata[index]
        outdata = np.zeros(needata.shape)
        for i in range(needata.shape[0]):
            a, b = np.nanpercentile(needata[i], (per, 100 - per))
            outdata[i] = np.clip(needata[i], a, b)
            outdata[i] = (outdata[i] - a) / (b - a)
        if savepath is None:
            return outdata
        else:

            rgb=reshape_as_image(outdata)
            plt.figure()
            ax=plt.imshow(rgb)
            ax.axes.axis(False)
            plt.tight_layout()
            plt.savefig(savepath,format='jpg',dpi=500,pad_inches=0.05,bbox_inches='tight')
            plt.show()
    def is_contain_chinese(self,check_str):
        """
        判断字符串中是否包含中文
        :param check_str: {str} 需要检测的字符串
        :return: {bool} 包含返回True， 不包含返回False
        """
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                # print(1)
                return True
        return False
    def plot_show(self,data,norm,danwei,cmap,rgbdata=None,date=None):
        f, ax = plt.subplots(1, 1)
        # norm = plt.Normalize(0, 40)
        
        if addrgbdata==True:
            if rgbdata is not None:
                plt.imshow(rgbdata)  
        axa = plt.imshow(data, cmap, norm=norm)
        axa.axes.axis(False)
        bar = plt.colorbar(axa,shrink=0.68)
        # bar=f.colorbar(ax)

        if self.is_contain_chinese(danwei):
            bar.set_label(danwei, family='Simsun')
        else:
            bar.set_label(danwei, family='Times new Roman')
        [s.set_family('Times new Roman') for s in bar.ax.get_yticklabels()]
        if date is not None:
            plt.title(date,family='Times new Roman')
        plt.margins()
        plt.tight_layout()
        # plt.show()
        return f,ax
    def postclasscope(self,class_predictiona, removeholethresh=100, removesmallobject=50):

        """
        可以减少小物体,#该方法是利用skiamge里面的内容，主要有两部分，移除空洞以及小物体，该方法更好
        """
        arr_gb = pd.Series(class_predictiona.ravel())
        arr_gb = arr_gb.value_counts()
        arr_gb.sort_index(inplace=True)
        rsaa = list(arr_gb.items())
        dsaay = np.zeros([class_predictiona.shape[0], class_predictiona.shape[1], len(list(rsaa))])
        for t, (s, r) in enumerate(rsaa):  # 存在问题

            if s == 0:
                # print(s)
                bi = np.ones(class_predictiona.shape) * (class_predictiona == s)
                # print(np.sum(bi) == r)
                bi = np.bool_(bi)
                bi = morphology.remove_small_objects(bi, min_size=removesmallobject, connectivity=1)
                dsaay[:, :, t] = bi * (s + 30)
            else:
                bi = np.ones(class_predictiona.shape) * (class_predictiona == s)
                bi = np.bool_(bi)
                bi = morphology.remove_small_objects(bi, min_size=removesmallobject, connectivity=1)
                binary = morphology.remove_small_holes(bi, area_threshold=removeholethresh, connectivity=1)
                dsaay[:, :, t] = binary * s
        rsar = np.array(np.sum(dsaay, axis=2))
        rsar[rsar == 0] = np.nan
        # rsar[rsar == 30] = 0
        rsar[rsar > np.nanmax(arr_gb.keys())] = np.nan
        return rsar

class calbydate:
    def __init__(self,files):
        self.files=files
        self.data =None

    def fenleibydate(self, freq=None):
        fenlei = {}
        # print(self.files)
        dates = [re.search(r"(\d{4}\d{1,2}\d{1,2})", os.path.basename(file)).group(0) for file in  self.files]
        # dates=[date.group() for date in dates]
        needaa = pd.PeriodIndex(dates, freq=freq)
        for Q, file in zip(needaa,  self.files):
            data = fenlei.setdefault(Q.strftime('%Y-%m'), [])
            # date = datetime.datetime.strptime(file.split('\\')[-1][0:8], '%Y%m%d')
            data.append(file)
        return fenlei
    def readdata(self,file):
        self.src=rasterio.open(file)
        return self.src
    def writeout(self, outfile,ms,src=None):
        import rasterio
        if src is None:
            src=self.src
        # ms=np.where(ms<0,np.nan,ms)
        with rasterio.open(outfile, mode='w', driver='GTiff',
                           width=src.width, height=src.height, count=ms.shape[0],
                           crs=src.crs, transform=src.transform, dtype=ms.dtype) as dst:
            for s in range(ms.shape[0]):
                dst.write(ms[s], s + 1)
            dst.close()
    def caldata(self,nedfiles):
        # length=len(nedfiles)
        dataneed=[]
        for fe in nedfiles:
            # print(fe)
            data=self.readdata(fe).read()
            data=np.where(np.isinf(data),np.nan,data)
            mean=np.nanmean(data)
            # if fe=='K:\工作\蓝丰\云南九湖/泸沽湖\Product/Chla\20230626T035539_20230626T041013_T47_Chla.tif':
            #     print('data')
            print(data.shape)
            print(f'{fe}均值为{np.round(mean,2)}')
            if len(list(data.shape))>2:
                dataneed.append(data)
            else:
                dataneed.append(np.expand_dims(data,axis=0))
        dataneed=np.concatenate(dataneed)
        return dataneed
    def get_data(self,fenlei=None,keys=None):
        # fenlei=self.fenleibydate(freq)
        data={}
        if keys is None:
            for key in fenlei.keys():
                nedfiles=fenlei[key]
                dataneed=self.caldata(nedfiles)
                data[key]=dataneed
        else:
            for key in keys:
                nedfiles = fenlei[key]
                dataneed = self.caldata(nedfiles)
                data[key] = dataneed
        return data
    def show_data(self,product='SPM',freq=None,need=None,outdir=None,Allinone=True):
        # product以及freq是必要的
        """product 参数表示产品名称，默认为 "SPM"。
freq 参数表示频率，默认为 None。
need 参数表示需要展示的数据集合，默认为 None。
outdir 参数表示输出文件夹路径，默认为 None。
Allinone 参数表示是否将所有数据集合在一起展示。当其值为 True 时，展示所有数据；否则，
根据 outdir 参数指定的输出路径分别展示不同的数据集,包括图像和数据。
该方法首先设置 matplotlib 的字体大小为 12，并初始化一个空字典 outdata。
根据 freq 参数获取分类后的数据，并通过 get_data() 方法获取需要展示的数据集。
如果 Allinone 参数为 True，则将数据集合并在一起展示，并返回展示结果和 fig 对象。
否则，将不同的数据集分别展示，并保存到指定的输出路径中。
在展示数据时，将数据中大于 500 的值替换为 NaN，并计算每行的平均值，最后使用颜色映射 cmap 和归一化器 norm 将数据可视化。
最后，将颜色条添加到展示结果中，并输出到指定的输出路径或者屏幕上。"""
        import matplotlib as mpt
        mpt.rcParams['font.size'] = 12
        outdata={}
        fenlei = self.fenleibydate(freq)
        if need is None:
            needdata = self.get_data(fenlei)
        else:
            needfenlei={}
            for k in  need:
                
                needfenlei[k]=fenlei[k]
            # needfenlei=[fenlei[k] for k in need]
            needdata = self.get_data(needfenlei)
        if Allinone==True:
            lens = len(needdata.keys())
            cols = int(np.sqrt(lens))
            fig, axs = plt.subplots(cols, int(np.ceil(lens / cols)))
            for r,da in enumerate(list(needdata.keys())):
                try:
                    ax = axs.ravel()[r]
                except:
                    ax = axs
                data = needdata[da]
                data = np.where(data > 500, np.nan, data)

                # onedata=np.nanmedian(data,axis=0)
                onedata = np.nanmean(data, axis=0)
                outdata[da]=onedata
                danwei=dataplot[product]['dw']
                norm=dataplot[product]['norm']
                cmap=dataplot[product]['cmap']
                try:
                    norm=getnorm(product,onedata)
                except:
                    pass
                
                if medianyn:
                    
                    onedata = median(onedata, disk(5))
                    onedata=calindex(1,1).show(onedata,2)
                    # onedata=np.expand_dims(onedata,axis=0)
                try:
                    # norm=getnorm(product,data)
                    means=np.nanmean(onedata)
                    cmap,norm=getnorm_cmap(product,means)
                except:
                    pass
                # norm = plt.Normalize(0, 40)
                axa = ax.imshow(onedata, cmap, norm=norm)
                axa.axes.axis(False)
                # if product=='vb_fah':
                #     da='水华指数'
                #     family = 'Simsun'
                # else:
                family = 'Times new roman'
                ax.set_title(da,family=family)
                # bar=f.colorbar(ax)
            # plt.subplots_adjust(wspace=0,
            #                     hspace=0)
            fig.tight_layout()
            # cax = fig.add_axes([0,0,1,0.01])
            bar = fig.colorbar(axa,extend='both',orientation='horizontal',ax=axs,pad=0,shrink=0.68)
            bar.set_label(danwei, family='Times new roman')
            [s.set_family('Times new Roman') for s in bar.ax.get_xticklabels()]
            plt.margins(0,0)
            # plt.show()
            return outdata,fig
        else:
            for r, da in enumerate(list(needdata.keys())):
                fig, ax = plt.subplots(1 ,1)
                data = needdata[da]
                if  product not in ['SPM','Chla']:
                    data = np.where(data > 500, np.nan, data)
                else:
                    data = np.where(data > 500, 500, data)
                # onedata=np.nanmedian(data,axis=0)
                onedata = np.nanmean(data, axis=0)
                # print(np.nanmean(onedata))
                # outdata[da] = onedata
                danwei = dataplot[product]['dw']
                norm = dataplot[product]['norm']
                cmap = dataplot[product]['cmap']
                try:
                    norm=getnorm(product,onedata)
                except:
                    pass
                if medianyn:
                    onedata = median(onedata, disk(5))
                    onedata=calindex(1,1).show(onedata,2)
                    # onedata=np.expand_dims(onedata,axis=0)
                try:
                    # norm=getnorm(product,data)
                    means=np.nanmean(onedata)
                    cmap,norm=getnorm_cmap(product,means)
                except:
                    pass
                # norm = plt.Normalize(0, 40)
                axa = ax.imshow(onedata, cmap, norm=norm)
                axa.axes.axis(False)
                bar = plt.colorbar(axa,shrink=0.68)
                # bar=f.colorbar(ax)
                plt.title(da,family='Times new roman')
                if product in ['vb_fah','ndvi']:
                    bar.set_label(danwei, family='Simsun')
                    [s.set_family('Times new Roman') for s in bar.ax.get_yticklabels()]
                else:
                    bar.set_label(danwei, family='Times new roman')
                    [s.set_family('Times new Roman') for s in bar.ax.get_yticklabels()]
                plt.margins()
                plt.tight_layout()
                if outdir is not None:
                    outfile=outdir+product+'_'+da+'.jpg'
                    if os.path.exists(outfile):
                        continue
                    # plt.show()
                    plt.savefig(outfile, format='jpg', dpi=500, pad_inches=0.05, bbox_inches='tight')
                    outtiff=outdir+product+'_'+da+'.tif'
                    print('输出tif路径',outtiff)
                    self.writeout(outtiff, np.expand_dims(onedata, axis=0))

class ReadGoci:
    def __init__(self,file):
        self.file=file
    def read(self):
        ds = Dataset( self.file)
        file = ds.ncattrs()
        data = {}
        for gr in ds.groups:
            # print(ds[gr])
            # print(ds['geophysical_data']['RhoC'])
            if gr == 'geophysical_data':
                for dsa in ds['geophysical_data']['RhoC'].variables.keys():
                    # print(dsa)
                    at = ds['geophysical_data']['RhoC'].variables[dsa][:]
                    data[dsa] = at
                    # plt.figure()
                    # plt.imshow(at[1700:2000,950:1250],'jet',vmax=0.15)
                    # plt.title(dsa)
                    # plt.show()
        data = [data[a] for a in sorted(list(data.keys()))]
        #
        datay = np.concatenate([np.expand_dims(s, axis=0) for s in data], axis=0)
        return datay
    def readlonlat(self):
        ds = Dataset(self.file)
        lon=np.array(ds['/navigation_data/longitude'][:])
        lat=np.array(ds['/navigation_data/latitude'][:])
        return lon,lat

