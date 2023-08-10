# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: cope.py
 @time: 2023/4/19 16:17
 @productname: langfeng
 @说明：
'''

import os
# os.environ['PROJ_LIB'] = r'D:\CONDA\hsi_classfication\Lib\site-packages\osgeo\data\proj'
# os.environ['GDAL_DATA'] = r'D:\CONDA\hsi_classfication\Lib\site-packages\osgeo\data'
import glob
import geopandas as gpd

import zipfile
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import os
from tqdm import tqdm
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.geometry import Point

def getallfile(putzipdir,need):
    needfilse=[]
    for dirpath, dirnames, filenames in os.walk(putzipdir):
        for file in filenames:
            px=os.path.splitext(file)[1]
            if px==need:
                filea=os.path.join(dirpath,file)
                if os.path.exists(filea):
                    if filea not in needfilse:
                        needfilse.append(filea)
    return needfilse




class cope_S2_from_GEE:
    def __init__(self,putzipdir,bandcounts):
        self.bandcounts=bandcounts
        self.putzipdir=putzipdir
        self.tiffiles=getallfile(putzipdir,'.tif')
        self.csvfiles=getallfile(putzipdir,'.csv')

    def unzip(self,zipfiles):
        if not os.path.exists(self.putzipdir):
            os.mkdir(self.putzipdir)
        for file in zipfiles:
            # print(file)
            try:
                if zipfile.is_zipfile(file):
                    z = zipfile.ZipFile(file, "r")
                    try:
                        dir=set([s.split('/')[0] for s in z.namelist()])
                    except:
                        dir=['None','None']
                    if len(dir)==1:
                        unzipoutdirl=self.putzipdir
                    else:
                        unzipoutdirl = os.path.join(self.putzipdir, os.path.split(file)[1].split('.')[0])
                    outfile = os.path.join(unzipoutdirl, z.namelist()[0].split('/')[0])
                    if not os.path.exists(unzipoutdirl):
                        os.mkdir(unzipoutdirl)
                    if os.path.exists(outfile):
                        continue
                    try:
                        z.extractall(unzipoutdirl)
                        z.close()
                    except:
                        import subprocess
                        command = ['D:\Program Files\WinRAR\WinRAR.exe', 'x', '-o+']
                        command.extend([file, self.putzipdir])
                        subprocess.run(command, check=True)
                else:
                    continue
            except:
                pass  # 加不加都无所谓，但是加了可以跳过这个出现错误的代码块
            continue
        re = []  # 从zip文件中直接获取存在的数据
        for file in zipfiles:
            # print(file)
            try:
                if zipfile.is_zipfile(file):
                    z = zipfile.ZipFile(file, "r")
                    re = re + z.namelist()
            except:
                print(file, '存在问题')
                continue
        return re

    def quchong(self):
        tif = []
        tiffilea = []
        for s in self.tiffiles:
            name = os.path.basename(s)
            if name not in tif:
                tif.append(name)
                tiffilea.append(s)
            else:
                os.remove(s)
                print('删除',s)
        return tiffilea

    def match_file(self):  # 效率较高，用于匹配两者数据
        csvdata = {}
        wrongfile = []
        for csv in self.csvfiles:
            if os.stat(csv).st_size < 800: #小于1kb
                # print(csv)
                wrongfile.append(csv)
                continue
            csvdata[csv] = pd.read_csv(csv).shape[0]
        tifdata = {}
        wrongfiletif = []
        for tif in self.tiffiles:
            if os.stat(tif).st_size < 1200:
                # print(tif)
                wrongfiletif.append(tif)
                continue
            tifdata[tif] = rasterio.open(tif).count / self.bandcounts
        self.hasddata = {}
        hasddatas = []
        allsize = []
        for tif in tqdm(tifdata):
            tifname = tif.split('\\')[-1].split('tif')[0]
            # tifname = tif.split('\\')[-1].split('-')[0][:-3]
            for cfile in csvdata:
                tifds = tifdata[tif]
                csvds = csvdata[cfile]
                csvname = cfile.split('\\')[-1][:-14]
                # print(tifname)
                if tifname == csvname and tifds == csvds:
                    # print(1)
                    if tif not in self.hasddata:
                        hasddatas.append(tifname)
                        ns = self.hasddata.setdefault(tifname, [])
                        ns.append(tif)
                        ns.append(cfile)
                        allsize.append(os.stat(tif).st_size)
                        allsize.append(os.stat(cfile).st_size)
        self.allsized = np.round(sum(allsize) / (1024 ** 3),3)  # 转为GB

        return self

    def writeout(self,outfile, src, ms):
        import rasterio
        # ms=np.where(ms<0,np.nan,ms)
        with rasterio.open(outfile, mode='w', driver='GTiff',
                           width=src.width, height=src.height, count=ms.shape[0],
                           crs=src.crs, transform=src.transform, dtype=ms.dtype,tiled=True, compress='lzw') as dst:
            for s in range(ms.shape[0]):
                dst.write(ms[s], s + 1)
            dst.close()

    def getdata(self,src, shape=None):
        ns = []
        r = 0
        wrongindex = []
        ds = src.read().astype(np.int_)#!读成为int
        # ds=np.where(ds<0,np.nan,ds)
        for s in range(shape):
            dsa = r + s * self.bandcounts
            ms = ds[dsa:dsa + self.bandcounts]
            # SMNDWIindex, SMNDWI=SMNDWIget(ms)
            # if ms.max()>10000 or ms.min() <-100:
            # print(s)
            # wrongindex.append(s)
            ns.append(ms)
        return ns, wrongindex
    def copecsvtif(self,tif, csv, outcsvdir):
        '''
        导出分割的Tif数据及CSV属性表格
        '''
        csvfilae = pd.read_csv(csv)
        shape = csvfilae.shape[0]
        src = rasterio.open(tif)
        # outcsvdir = outcsvdirname
        count=0
        for s in range(csvfilae.shape[0]):
                csva = pd.DataFrame(csvfilae.iloc[s]).T
                name = csva['system:index'].values[0]
                if os.path.exists(outcsvdir + name + '.csv') and os.path.exists(outcsvdir + name + '.tif'):
                    count+=1
                    

        if count!=shape:
            try:
                ns, wrongindex = self.getdata(src, shape=shape)
                for s in range(csvfilae.shape[0]):
                    if s in wrongindex:
                        continue
                    else:
                        csva = pd.DataFrame(csvfilae.iloc[s]).T
                        name = csva['system:index'].values[0]
                        if os.path.exists(outcsvdir + name + '.csv') and os.path.exists(outcsvdir + name + '.tif'):
                            continue
                        csva.to_csv(outcsvdir + name + '.csv', index=False)
                        file = outcsvdir + name + '.tif'
                        
                        outds=np.array(ns[s]).astype(np.float32)
                        outds=np.where((outds<-50)|(np.isinf(outds)),np.nan,outds)#!如果envi打不开则是因为出现无效值，比如说极大值
                        self.writeout(file, src, outds)
            except:
                print('无法导入数据')
                for s in range(csvfilae.shape[0]):
                    index = list(range((s) * self.bandcounts + 1, (s) * self.bandcounts + self.bandcounts + 1))
                    csva = pd.DataFrame(csvfilae.iloc[s]).T
                    name = csva['system:index'].values[0]
                    if os.path.exists(outcsvdir + name + '.csv') and os.path.exists(outcsvdir + name + '.tif'):
                        continue
                    csva.to_csv(outcsvdir + name + '.csv', index=False)
                    file = os.path.abspath(outcsvdir + name + '.tif')
                    self.writeout(file, src, src.read(index))
    
    def writetifwithcsv(self,outcsvdir):
        goodfile = []
        wrong = []
        hasddata=self.hasddata
        for file in tqdm(hasddata):
            print(file)
            try:
                if len(hasddata[file]) == 2:
                    tif, csv = hasddata[file]
                    tifds = rasterio.open(tif).count / self.bandcounts
                    csvds = pd.read_csv(csv).shape[0]
                    if tifds == csvds:
                        goodfile.append(tif)
                        outcsvdirname =os.path.join( outcsvdir , file) + '/'
                        if not os.path.exists(outcsvdirname):
                            os.mkdir(outcsvdirname)
                            self.copecsvtif(tif, csv, outcsvdirname)  # 导出分割的Tif数据及CSV属性表格
                    else:
                        wrong.append(file)
                    # wrong.append(csv)
                else:
                    for ri in range(len(hasddata[file]) // 2):
                        tif, csv = hasddata[file][ri * 2], hasddata[file][ri * 2 + 1]
                        print(tif)
                        tifds = rasterio.open(tif).count / self.bandcounts
                        csvds = pd.read_csv(csv).shape[0]
                        if tifds == csvds:
                            goodfile.append(tif)
                            name=os.path.splitext(tif.split('\\')[-1])[0]
                            outcsvdirname = os.path.join( outcsvdir , file)+ '/' +name + '/'
                            if not os.path.exists(outcsvdirname):
                                os.makedirs(outcsvdirname)
                            self.copecsvtif(tif, csv, outcsvdirname)  # 导出分割的Tif数据及CSV属性表格
                        else:
                            wrong.append(file)
            except:
                print(len(hasddata[file]))
                print(hasddata[file])

    def mosaic(self,outfile, alltiff):

        from osgeo import gdal

        # 创建选项对象
        options = gdal.WarpOptions(creationOptions=['COMPRESS=LZW','BIGTIFF=YES'],multithread = True)

        # 调用 gdal.Warp() 函数进行图像重投影
        gdal.Warp(outfile, alltiff, options=options)
    
    def rasteriomerge(self,dst_file,alltiff):
        # if not os.path.exists(dst_file):
        src_files_to_mosaic = []
        for tif_f in alltiff:
            src = rasterio.open(tif_f)
            src_files_to_mosaic.append(src)
        out_meta = src.meta.copy()
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "dtype":mosaic.dtype,
                    "tiled":True, 
                    "compress":'lzw'
                    }
                    )
        # 保存文件
        with rasterio.open(dst_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        """示例"""

    def mosaic_batch(self,src_folders, dst_folder):
        # src_folder1 = 'folder1'
        # src_folder2 = 'folder2'
        #
        # # 输出文件夹的路径
        # dst_folder = 'output'

        # 遍历文件夹1，找到名字相同的图像
        count=[]
        for file in src_folders:
            count.append(len(glob.glob(file+'/*.tif')))
        index=np.argmax(count)
        if np.sum(count)!=0:
            for filename in os.listdir(src_folders[index]):

                if  os.path.splitext(filename)[1] not in ['.tif']:
                    continue
                # 构造图像的路径
                alltiff = []
                for name in src_folders:

                    src_file1 = os.path.join(name, filename)
                    if os.path.exists(src_file1):
                        alltiff.append(src_file1)
                # 调用 gdal.Warp() 函数进行镶嵌
                dst_file = os.path.join(dst_folder, filename)
                if os.path.exists(dst_file):
                    print(dst_file+'已存在')
                    continue
                # self.mosaic(dst_file, alltiff)
                
                t1=time.time()
                # self.mosaic(dst_file, alltiff)#!32.40555214881897
                self.rasteriomerge(dst_file,alltiff)#!18.397470951080322
                t2=time.time()
                print(t2-t1)

    def Clipimage(self,indir,outdir,shpfile):

        files=glob.glob(indir+'/*.tif')
        geom=gpd.read_file(shpfile).geometry
        for file in files:
            filename=os.path.basename(file)
            outfile=os.path.join(outdir,filename)
            src=rasterio.open(file)
            feature = [mapping(geom)]
            try:
                out_image, out_transform = mask(src, geom, crop=True)
                out_meta = src.meta.copy()
                out_image=np.array(out_image).astype(np.float32)
                out_image=np.where((out_image<=0)|(np.isinf(out_image)),np.nan,out_image)#!将小于等于0或无限大的值设置为nan
                # mosaic, out_trans = merge(src_files_to_mosaic)
                out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "dtype":out_image.dtype,
                            "tiled":True, 
                            "compress":'lzw'
                            }
                            )
                # 保存文件
                with rasterio.open(outfile, "w", **out_meta) as dest:
                    dest.write(out_image)
            except:
                continue

        
# outdir='K:\遥感数据及处理\遥感数据\GEE\Sentinel-2\L2A\萧山区河' # !需要修改
# for name in ['zip','data','mosaic']:
#     dirs=os.path.join(outdir,name)
#     if not os.path.exists(dirs):
#         os.makedirs(dirs)

# zipfiles=glob.glob(f"{outdir}\zip/*.zip")
# putzipdir=f"{outdir}\zip"
# CopeS2=cope_S2_from_GEE(putzipdir,13)
# cope_S2_from_GEE(putzipdir,13).unzip(zipfiles) #!解压
# outcsvdir=rf"{outdir}\data/"
# matchdata=CopeS2.match_file().hasddata
# CopeS2.match_file().writetifwithcsv(outcsvdir) #!拆分
# for src_f in glob.glob(outcsvdir+"*"):
#     dst_folder=f'{outdir}/mosaic/'+os.path.basename(src_f)
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#     src_folders=glob.glob(src_f+'/*')
#     CopeS2.mosaic_batch(src_folders, dst_folder)#!合并融合
# import requests
# import time
# res = requests.get(
#             "http://wxpusher.zjiecode.com/demo/send/custom/UID_2jf05Kebz531rva3cQaL1Z8612qK?content={}".format(
#                 time.strftime('%Y-%m-%d %H:%M:%S') + '影像分割及融合完成'))
