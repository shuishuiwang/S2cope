# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: outputJPGmodule.py
 @time: 2023/6/1 10:12
 @productname: Code
 @说明：这个类的作用是实现一些遥感影像处理操作，如生成指数、计算水质参数等，并将结果输出为图片或文件。类中包含了五个方法：

outJPG_allindex_pro: 生成多种指数图像并保存为jpg格式。
outRsindex: 生成遥感图像的RS指数图像并保存为jpg格式。
outindexproone: 根据所选参数生成一个指数图像及其对应的tif文件，并将结果保存为jpg格式。
outRSproalljpg: 生成所有水质参数的图像并将结果保存为jpg格式。
outRSprotif: 根据所选水质参数生成一个tif文件，并将结果保存。
outRSprojpg: 根据所选水质参数生成一个jpg文件。
该类需要四个参数：src、Cal、path和name。其中，src 是遥感影像数据的路径；Cal 是一个自定义的类，包含一系列计算遥感图像指数和水质参数的方法；path 是生成结果文件的保存路径；name 是生成结果文件名的前缀。
在类初始化时还定义了 needspm 和 needchla 参数，分别用于获取 SPM 和 Chla 水质参数。
'''
from rasterio.plot import reshape_as_image
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import os
import glob
import rasterio
import gc
import sys
from Config import *
sys.path.append("K:\Project_engpath\WaterqualityDL\Out")
sys.path.append("K:\个人代码工具\Module")

Wqdeepl=__import__("Wqdeepl")
gettestdata=Wqdeepl.gettestdata

from Wqdeepl import *
from Config import *
calm = __import__('cal_RSindex').calindex
Cbydate = __import__('cal_RSindex').calbydate
CopeS2 = __import__('Cope_Data_from_GEE').cope_S2_from_GEE

class OutJpgAllIndexPro:
    # medianyn=True
    needspm = ['6_GWL硕士论文(560-665)/(560+665)']  # SPM还可以的
    needchla = ['9_平寨水库(1/665-1/705)*842']  # chal还可以的
    def __init__(self, src, Cal, path, name):
        self.src = src  # src = rasterio.open(file)
        self.Cal = Cal  # Cal = calm(data, wavelength)
        self.path = path  # path='K:\工作\蓝丰\滇池/'
        self.name = name  # name=os.path.basename(os.path.splitext(file)[0])
        self.date=re.findall('\d{8}',os.path.split(src.name)[1])[0]
        self.outrasterorjpgsets = {
            'Rsindex': {'hanshu': self.outRsindex, 'outdir': os.path.join(path, 'JPG'), 'need': ['rgb'], 'exc': True},# 导出所有指数产品在一张图 'need':['vb_fah','rgb']
            'indexpro': {'hanshu': self.outindexproone, 'outdir': os.path.join(path, 'Product'), 'need': ['vb_fah'],'exc': True},# 导出指数产品tif格式
            'RSproalljpg': {'hanshu': self.outRSproalljpg, 'outdir': os.path.join(path, 'JPG'), 'need': ['Chla'], 'exc': False},# 导出水质产品在一张图+['NTU','KMNO','NH3N','Trans','DO','TP', 'TN']
            'outRSprotif': {'hanshu': self.outRSprotif, 'outdir': os.path.join(path, 'Product'),   'need': ['SPM','Chla'], 'exc': True},# 导出水质产品tif格式
            'outRSprojpg': {'hanshu': self.outRSprojpg, 'outdir': os.path.join(path, 'Product'),'need': [ 'SPM','Chla'], 'exc': True},# 导出水质产品jpg格式
        }

    def outJPG_allindex_pro(self):
        for name, vals in self.outrasterorjpgsets.items():
            if vals['exc']:
                vals['hanshu'](vals['outdir'], self.name, vals['need'])
        return self.outrasterorjpgsets

    def outRsindex(self, outdir, name, need=None):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = f"{outdir}/{name}_RSindex.jpg"
        if not os.path.exists(outfile):
            fig, allndex = self.Cal.getall_index(need=need)
            fig.savefig(outfile, format='jpg', dpi=500, pad_inches=0.05, bbox_inches='tight')
            plt.close(fig)

    def outindexproone(self, outdirall, name, need=None):
        if need is None:
            need = ['vb_fah']
        for CP in need:
            outdirs = f'{outdirall}/{CP}'
            if not os.path.exists(outdirs):
                os.makedirs(outdirs)
            outfftif = f"{outdirs}/{name + '_' + CP}.tif"
            outfile = f"{outdirs}/{name + '_' + CP}.jpg"
            if not os.path.exists(outfftif):
                
                fig, allndexnew = self.Cal.getall_index(need=[CP],date=self.date)
                # ax.set_label(self.date)
                # fig.show()
                fig.savefig(outfile, format='jpg', dpi=1000, pad_inches=0.05, bbox_inches='tight')
                plt.close(fig)
                da = allndexnew[CP]
                if medianyn:
                    da=self.Cal.show(da,1)
                    da = median(da, disk(3))
                CopeS2('1', '1').writeout(outfile=outfftif, src=self.src, ms=np.expand_dims(da, axis=0))
                print(f'导出tif文件{outfftif}')
    def outRSproalljpg(self, outdir, name, need=None):
        if need is None:
            need = ['Chla', 'SPM']
        for pro in need:
            outfile = f"{outdir}/{name}_{pro}_allpro.jpg"
            if not os.path.exists(outfile):
                allpro = self.Cal.calpro(pro)
                fig2 = self.Cal.getshowall_WQ(outpro=allpro)
                fig2.savefig(outfile, format='jpg', dpi=500, pad_inches=0.05, bbox_inches='tight')
                plt.close(fig2)
                
    def get_zwname(self,product):
        if product == 'TN':
            return '总氮'
        
        elif product == 'TP':  
            return '总磷'
        
        elif product == 'NTU':
            return '浊度'  

        elif product == 'KMNO':
            return '高锰酸盐指数'
        
        elif product == 'NH3N':
            return '氨氮'

        elif product == 'Trans':
            return '透明度（cm）'
        elif product == 'DO':
            return '溶解氧'
        else:
            return None
    def outRSprotif(self, outdir, name, need=None):
        if need is None:
            need = ['Chla', 'SPM','TN']
        for product in need:
            if not os.path.exists(f'{outdir}/{product}'):
                os.makedirs(f'{outdir}/{product}')
            outfftif = f"{outdir}/{product}/{name}_{product}.tif"
            if not os.path.exists(outfftif):
                if product == "SPM":
                    if self.Cal.iszaohua():
                        outda = self.Cal.sertmodel()/2
                    else:
                        data = self.Cal.calpro(product, self.needspm)
                        da = np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()])
                        QAall = self.Cal.ndwi() * self.Cal.QAforS2()
                        outda = da * QAall
                        # outda = self.Cal.ndvi() * QAall
                        # matplotlib.use('Tkagg')
                        # plt.imshow(outda[0],'jet')
                        # plt.show()
                        
                        
                elif product == "Chla":
                    # data = self.Cal.calpro(product, self.needchla)
                    # da = np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()])
                    # QAall = self.Cal.ndwi() * self.Cal.QAforS2()
                    # outda = da * QAall
                    outda=self.Cal.mosaic_chla()
                    
                elif product in ['TN', 'TP', 'NTU', 'KMNO', 'NH3N', 'Trans','Chla','DO']:
                    if machinelearning:
                        zwname = self.get_zwname(product)
                        modeltype = parraset[zwname]['model']
                        outda = gettestdata(self.src,self.Cal, zwname, outmodeldir, SP=False, 
                                            trainMean=False, selectmod=modeltype)
                    
                if product == "Chla":
                    if self.Cal.iszaohua():#!表明存在藻华   
                        norm = plt.Normalize(0,100)
                        # bz=np.nanmean(outda)
                        # outda=outda
                if medianyn:
                    outda=self.Cal.show(outda,2)
                    outd = median(outda[0], disk(3))
                    outda=np.expand_dims(outd,axis=0)
                CopeS2('1', '1').writeout(outfile=outfftif, src=self.src, ms=outda)
                print(f'导出tif文件{outfftif}')
                del outda
                gc.collect()

    def outRSprojpg(self, outdir, name, need=None):
        date=re.findall('\d{8}',os.path.basename(self.src.name))[0]
        if need is None:
            need = ['Chla', 'SPM','TN']

        for product in need:
            danwei = dataplot[product]['dw']
            norm = dataplot[product]['norm']
            cmap = dataplot[product]['cmap']

            if not os.path.exists(f'{outdir}/{product}'):
                os.makedirs(f'{outdir}/{product}')
            outffchan = f"{outdir}/{product}/{name + '_' + product}.jpg"
            if not os.path.exists(outffchan):
                outtif=outffchan.replace('jpg','tif')
                if os.path.exists(outtif):
                    print(f'读取文件{outtif}')
                    data=rasterio.open(outtif).read()
                else:
                    if product == "SPM":
                        if self.Cal.iszaohua():
                            data = self.Cal.sertmodel()/2
                            norm =plt.Normalize(0, 150)
                        else:
                            data = self.Cal.calpro(product, self.needspm)
                            da = np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()])
                            QAall = self.Cal.ndwi() * self.Cal.QAforS2()
                            data = da * QAall
                    elif product == "Chla":
                        # data = self.Cal.calpro(product, self.needchla)
                        # da = np.concatenate([np.expand_dims(data[s], axis=0) for s in data.keys()])
                        
                        # QAall = self.Cal.ndwi() * self.Cal.QAforS2()
                        # data = QAall * da
                        
                        data=self.Cal.mosaic_chla()
                    elif product in ['TN', 'TP', 'NTU', 'KMNO', 'NH3N', 'Trans','Chla','DO']:
                        if machinelearning:
                            zwname = self.get_zwname(product)
                            modeltype = parraset[zwname]['model']
                            data = gettestdata(self.src,self.Cal, zwname, outmodeldir, SP=False, 
                                                trainMean=False, selectmod=modeltype)

                    if product == "Chla":
                        if self.Cal.iszaohua():#!表明存在藻华   
                            norm = plt.Normalize(0,100)
                            # bz=np.nanmean(data)
                            # data=data
                if medianyn:
                    outda = median(data[0], disk(3))
                    outda=self.Cal.show(outda,2)
                    data=np.expand_dims(outda,axis=0)
                try:
                    # norm=getnorm(product,data)
                    # np.nanmean(data)/np.nanstd(data)
                    means=np.nanmean(data)
                    cmap,norm=getnorm_cmap(product,means)
                    
                    if means>30:
                        if np.nanpercentile(data,90)-np.nanpercentile(data,10)<20:#!如果85分位数减去10分位小于10，则设置norm为None
                            norm=None
                    # norm=None
                except:
                    pass
                if data.shape[0] > 1:
                    print("数组纬度大于2，无法绘制")
                else:
                    if addrgbdata==True:
                        rgbdata=self.Cal.getrgbdata(savepath=None)
                    fig3, ax = self.Cal.plot_show(data[0], norm, danwei, cmap,rgbdata=rgbdata,date=date)
                    fig3.savefig(outffchan, format='jpg', dpi=500, pad_inches=0.05, bbox_inches='tight')
                    plt.close(fig3)
                del fig3
                del data
                gc.collect()
                

class Outprobydate:
    # needpro=[]
    # needpro = ['Chla', 'SPM', 'vb_fah']  # 设置导出的产品

    def __init__(self, path):
        self.path = path
        self.indir = os.path.join(path, 'Product')
        self.needpro=os.listdir(self.indir)
    def out_tifjpg_byfreq(self, freq=None, needate=None, needpro=None, Allinone=False):
        
        """按照给定的频率（默认为月）对数据进行分类和处理，输出结果为tif和jpg格式文件，
        对于Allinone参数的不同取值，导出不同形式的图像文件
        self: 表示类实例本身。
        freq=None: 表示数据处理和导出时使用的频率，默认为'M'（按月）。
        needate=None: 表示需要处理和导出的日期列表，默认为None，即使用所有日期。
        needpro=None: 表示需要处理和导出的产品列表，默认为类属性self.needpro。
        Allinone=False: 控制是否将所有日期的数据导出到一张图中，默认为False，即导出多张图片。
        """
        if freq is None:
            freq = 'M'  # 按月
        outprodata = {}
        if needpro is None:
            needpro=self.needpro
        # 针对产品列表中的每个产品进行操作
        for product in needpro:
            dst_folder = self.indir + f"/{product}/" # 获取产品文件夹路径
            files = glob.glob(dst_folder + f'/*{product}.tif') # 获取指定产品的所有tif文件
            Cbd = Cbydate(files) # 将tif文件按照日期进行分类
            outdir = self.indir + f"/{product}/" 
            
            # 如果输出目录不存在，则创建一个新的目录
            if not os.path.exists(outdir):
                print('创建：'+outdir)
                os.makedirs(outdir)
                
            fenlei = Cbd.fenleibydate(freq) # 按照给定频率对数据进行分类
            # if needate is None:
            needatey = list(fenlei.keys())
            if needate is not None:
                needatey=needate
            # print(needatey)
                
            # 根据Allinone参数的不同取值，导出不同形式的图像文件
            if Allinone == False:
                Cbd.show_data(product=product, freq=freq, need=needatey, outdir=outdir,
                            Allinone=Allinone)  # ,Allinone=False导出jpg和tif,Allinone=True导出一张图
            else:
                outfile = os.path.join(outdir, product, 'Allinone'+ '.jpg')
                if not os.path.exists(outfile):
                    outdata, fig = Cbd.show_data(product=product, freq=freq, need=needatey, outdir=outdir,
                                                Allinone=Allinone)  # ,Allinone=False导出jpg和tif,Allinone=True导出一张图
                    plt.savefig(outfile, format='jpg', dpi=500, pad_inches=0.05, bbox_inches='tight')
                    plt.close(fig)
                    # outprodata[product]=outdata
                    # return fig
