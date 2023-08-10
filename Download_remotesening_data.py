import pandas as pd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import pyautogui
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pyperclip
import time
import requests
import numpy as np
from osgeo import ogr
import json
from sentinelsat import SentinelAPI
import os
import geopandas as gpd
from tqdm import tqdm
from subprocess import call
from bs4 import BeautifulSoup as bs
import glob
from selenium.webdriver.common.by import By
import zipfile
import datetime
from Cope_Data_from_GEE import *
global IDM
IDM=r"E:\idm绿色直装\ucbug.com-idm\idm 6.39.2.2\IDMan.exe"

password={
    '1':['shuishui', 'shijianbuyu1234'],  #wang15982458096
    '2':['fayelyy123', 'lyy123456'],
    '3':['shuishui192', 'shijianbuyu1234'],  #1926728
    '4':['shuishui223', 'shijianbuyu1234'],
    '5':['shuishui512', 'shijianbuyu1234'],
    '6':['shuishui191', 'shijianbuyu1234'],    #wang19121099
}


class downfiles:
    api = SentinelAPI('shuishui', 'shijianbuyu1234', 'https://scihub.copernicus.eu/dhus')
    api1 = SentinelAPI('fayelyy123', 'lyy123456', 'https://scihub.copernicus.eu/dhus')
    api2 = SentinelAPI('shuishui192', 'shijianbuyu1234', 'https://scihub.copernicus.eu/dhus')
    # api3 = SentinelAPI('shuishui192', 'shijianbuyu1234', 'https://scihub.copernicus.eu/dhus')
    api4 = SentinelAPI('shuishui223', 'shijianbuyu1234', 'https://scihub.copernicus.eu/dhus')
    def __init__(self,startdate,enddate):
        self.startdate=startdate
        self.enddate=enddate
        # self.sensor=sensor
    # 	startdate:'YYYYMMDD'
    # 	enddate:'YYYYMMDD'
    def downloadlinksS2(self, geometry, percent,protype='S2MSI2A'):

        # api = SentinelAPI('shuishui', 'shijianbuyu1234', 'https://scihub.copernicus.eu/dhus')
        def acitvateS2links( S2link):
            needactivate = []
            onlineindexs = []
            for link in list(S2link.index):
                pd = self.api.is_online(link)
                if pd == True:
                    onlineindexs.append(link)
                else:
                    needactivate.append(link)
            return onlineindexs, needactivate
        """
        Sentinel-1: SLC, GRD, OCN
Sentinel-2: S2MSI2A,S2MSI1C, S2MS2Ap
Sentinel-3: SR_1_SRA___, SR_1_SRA_A, SR_1_SRA_BS, SR_2_LAN___, OL_1_EFR___, OL_1_ERR___, OL_2_LFR___,
 OL_2_LRR___, SL_1_RBT___, SL_2_LST___, SY_2_SYN___, SY_2_V10___, SY_2_VG1___, SY_2_VGP___, SY_2_AOD__,
  SL_2_FRP__.
Sentinel-5P: L1B_IR_SIR, L1B_IR_UVN, L1B_RA_BD1, L1B_RA_BD2, L1B_RA_BD3, L1B_RA_BD4, L1B_RA_BD5,
L1B_RA_BD6, L1B_RA_BD7, L1B_RA_BD8, L2__AER_AI, L2__AER_LH, L2__CH4, L2__CLOUD_, L2__CO____, L2__HCHO__,
L2__NO2___, L2__NP_BD3,
 L2__NP_BD6, L2__NP_BD7, L2__O3_TCL, L2__O3____, L2__SO2___, AUX_CTMFCT, AUX_CTMANA.
        """
        query_kwargs = {  # 查询的参数
            "platformname": 'Sentinel-2',
            "date": (self.startdate, self.enddate),
            "producttype": protype,
            # "sensoroperationalmode" : 'IW',
            # "orbitnumber" : 16302,
            # "relativeorbitnumber" : 130,
            # "orbitdirection" : 'ASCENDING',
        }

        def records(file):
            # generator
            reader = ogr.Open(file)
            layer = reader.GetLayer(0)
            for i in range(layer.GetFeatureCount()):
                feature = layer.GetFeature(i)
                yield json.loads(feature.ExportToJson())

        def getjson(aoi_geojson_fp):

            # aoi_geojson_fp = r'C:\wate quality\矢量\SITEStimewithfirst2.shp'
            ns = gpd.read_file(aoi_geojson_fp).to_json()
            geometrys = json.loads(ns)['features'][0]['geometry']
            # geojson_to_wkt(geometrys)
            return geometrys

        aoi_geojson_fp = geometry
        filepath, auxfile = os.path.splitext(aoi_geojson_fp)
        if auxfile == '.shp':
            try:
                footprint = geojson_to_wkt(records(aoi_geojson_fp))  # 读取geojson文件，获取足迹
            except:
                footprint = geojson_to_wkt(getjson(aoi_geojson_fp))  # 读取geojson文件，获取足迹

        elif auxfile == '.geojson':
            footprint = geojson_to_wkt(read_geojson(aoi_geojson_fp))  # 读取geojson文件，获取足迹
        else:
            print('not this file format')
        kw = query_kwargs.copy()
        try:
            results = self.api.query(footprint, **kw)

            df = self.api.to_dataframe(results)
            if 'gmlfootprint' in df.keys():
                df.pop('gmlfootprint')
            cloudcover = df['cloudcoverpercentage']
            index = np.flatnonzero(cloudcover <= float(percent))
            da = df.iloc[index, :]
            onlineindexs, needactivate=acitvateS2links(da)
            df1 =  da.loc[onlineindexs].assign(online=np.repeat([True],len(onlineindexs)))
            df2 =  da.loc[needactivate].assign(online=np.repeat([False],len(needactivate)))

            return pd.concat([df1,df2])

        except:
            name = os.path.basename(aoi_geojson_fp).replace('geojson', 'txt')
            # with open('..\\sup_load_geometry\\'+name,'w') as f:
            #     f.write('数据量太大，下载错误')
            return '数据量太大或者网络问题'
    
    def downloadlinksS3(self, geometry, protype='OL_1_EFR___'):

        # api = SentinelAPI('shuishui', 'shijianbuyu1234', 'https://scihub.copernicus.eu/dhus')
        def acitvateS3links( S3link):
            needactivate = []
            onlineindexs = []
            for link in list(S3link.index):
                pd = self.api.is_online(link)
                if pd == True:
                    onlineindexs.append(link)
                else:
                    needactivate.append(link)
            return onlineindexs, needactivate
        """
        Sentinel-1: SLC, GRD, OCN
Sentinel-2: S2MSI2A,S2MSI1C, S2MS2Ap
Sentinel-3: SR_1_SRA___, SR_1_SRA_A, SR_1_SRA_BS, SR_2_LAN___, OL_1_EFR___, OL_1_ERR___, OL_2_LFR___,
 OL_2_LRR___, SL_1_RBT___, SL_2_LST___, SY_2_SYN___, SY_2_V10___, SY_2_VG1___, SY_2_VGP___, SY_2_AOD__,
  SL_2_FRP__.
Sentinel-5P: L1B_IR_SIR, L1B_IR_UVN, L1B_RA_BD1, L1B_RA_BD2, L1B_RA_BD3, L1B_RA_BD4, L1B_RA_BD5,
L1B_RA_BD6, L1B_RA_BD7, L1B_RA_BD8, L2__AER_AI, L2__AER_LH, L2__CH4, L2__CLOUD_, L2__CO____, L2__HCHO__,
L2__NO2___, L2__NP_BD3,
 L2__NP_BD6, L2__NP_BD7, L2__O3_TCL, L2__O3____, L2__SO2___, AUX_CTMFCT, AUX_CTMANA.
        """
        query_kwargs = {  # 查询的参数
            "platformname": 'Sentinel-3',
            "date": (self.startdate, self.enddate),
            "producttype": protype,
            # "sensoroperationalmode" : 'IW',
            # "orbitnumber" : 16302,
            # "relativeorbitnumber" : 130,
            # "orbitdirection" : 'ASCENDING',
        }

        def records(file):
            # generator
            reader = ogr.Open(file)
            layer = reader.GetLayer(0)
            for i in range(layer.GetFeatureCount()):
                feature = layer.GetFeature(i)
                yield json.loads(feature.ExportToJson())

        def getjson(aoi_geojson_fp):

            # aoi_geojson_fp = r'C:\wate quality\矢量\SITEStimewithfirst2.shp'
            ns = gpd.read_file(aoi_geojson_fp).to_json()
            geometrys = json.loads(ns)['features'][0]['geometry']
            # geojson_to_wkt(geometrys)
            return geometrys

        aoi_geojson_fp = geometry
        filepath, auxfile = os.path.splitext(aoi_geojson_fp)
        if auxfile == '.shp':
            try:
                footprint = geojson_to_wkt(records(aoi_geojson_fp))  # 读取geojson文件，获取足迹
            except:
                footprint = geojson_to_wkt(getjson(aoi_geojson_fp))  # 读取geojson文件，获取足迹

        elif auxfile == '.geojson':
            footprint = geojson_to_wkt(read_geojson(aoi_geojson_fp))  # 读取geojson文件，获取足迹
        else:
            print('not this file format')
        kw = query_kwargs.copy()
        try:
            results = self.api.query(footprint, **kw)
            # self.api.download_all(results)
            df = self.api.to_dataframe(results)
            # df.to_csv('data.csv')
            if 'gmlfootprint' in df.keys():
                df.pop('gmlfootprint')
            onlineindexs, needactivate=acitvateS3links(df)
            df1 =  df.loc[onlineindexs].assign(online=np.repeat([True],len(onlineindexs)))
            df2 =  df.loc[needactivate].assign(online=np.repeat([False],len(needactivate)))
            return pd.concat([df1,df2])
        except:
            name = os.path.basename(aoi_geojson_fp).replace('geojson', 'txt')
            # with open('..\\sup_load_geometry\\'+name,'w') as f:
            #     f.write('数据量太大，下载错误')
            return '数据量太大或者网络问题'
    
    def writeoutlinks(self,links,outfile):
        with open(outfile,'w') as f:
            for link in links:
                f.write(link+'\n')
    def activateS2links(self,links):
        activaedlinks=[]
        for link in links:
            self.api.download(link)
            activaedlinks.append(link)
        return activaedlinks
    def downloadjpg(self,outpicdir,df):
        outpicdirs=os.path.join(outpicdir,'JPG')
        if not os.path.exists(outpicdirs):
            os.makedirs(outpicdirs)
        piclinks=df['link_icon']
        title=df['title']
        for piclink,name in zip(piclinks,title):
            # print(piclink,name)
            shuishui='shuishui'
            shijianbuyu1234='shijianbuyu1234'
            piclink=f"https://{shuishui}:{shijianbuyu1234}@"+piclink.split('https://')[1]
            if os.path.exists(outpicdirs+'/'+name+'.jpg'):
                continue
            cookie={
                'dhusAuth':'bbfbc7b1a1cbd446f6df47783e5702c3',
                'dhusIntegrity':'552f68e693e939e776a190a51b041fb855d57aae',
                'PPA_CI':'e2ee54201f5df02cbb5a5c2d61da262f',
            }
            ds=requests.get(piclink,cookies=cookie)
            with open(outpicdirs+'/'+name+'.jpg','wb') as f:
                f.write(ds.content)
            print(f'导出图片：{name}'+'.jpg')
                

    def downloadS3filebydriver(self,outdirs,df):
        outdir=outdirs+'/ZIP'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        delete_errorfile(outdir)#!删除未完成，同时删除下载多余的文件
        def is_file_in_use(file_path):
            t1=os.path.getsize(file_path)
            time.sleep(5)
            t2=os.path.getsize(file_path)
            print(t2-t1)
            try:
                os.remove(file_path)
                return False
            except IOError:
                return True
        def is_file_in_use_bysize(file_path):
            t1=os.path.getsize(file_path)
            time.sleep(15)
            t2=os.path.getsize(file_path)
            # print(t2-t1)
            if t2-t1>1000:
                # print(f'正在下载：{file_path}')
                return True
            else:
                return False
        
        def getcurrent_download_count(outdir):
            files=glob.glob(outdir+"/*.crdownload")
            count=0
            for file in files:
                try:
                    if is_file_in_use_bysize(file):
                        count+=1
                except:
                    continue
            print(f'当前下载个数为"{str(count)}"个')
            return count 
        try:
            changedownloaddir(outdir)#!根据浏览器配置文件修改下载地址和下载路径
            # outdir=r'C:\Users\SHUI\Downloads/'
            outpicdirs=outdir
            os.startfile(r'C:\Users\SHUI\Desktop\Google Chrome new.lnk')
            options = Options()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9527")
            driver = webdriver.Chrome(options=options)
        except:
            outpicdirs=os.path.join(outdir,'File')
            if not os.path.exists(outpicdirs):
                os.makedirs(outpicdirs)
            options = Options()
                    # 'profile.default_content_settings.popups': 0, 
            prefs = {'download.default_directory': os.path.abspath(outpicdirs)}
            options.add_experimental_option('prefs', prefs)
            driver = webdriver.Chrome(options=options)
        # print(driver.title)
        online=df['online']
        index=np.flatnonzero(online==True)
        dfonline=df.iloc[index]
        piclinks = dfonline['link']
        title = dfonline['title']
        offlineindex=np.flatnonzero(online==False)
        dfoffline=df.iloc[offlineindex]
        r = 0
        n = 0
        needdata=[]
        for piclink, name in zip(piclinks, title):
            outfile = os.path.join(outpicdirs, name) + '.zip'
            if not os.path.exists(outfile):
                if r % 3 == 0:
                    n += 1
                    shuishui, shijianbuyu1234 = password[str(n)]
                    # print(n,shuishui)
                    if n==6:
                        n=0#!如果n等于6的时候，则归0
                piclink2 = f"https://{shuishui}:{shijianbuyu1234}@" + piclink.split('https://')[1]
                # print(n,piclink2)
                driver.get(piclink2)
                needdata.append(piclink2)
                driver.get('chrome://downloads/')
                driver.implicitly_wait(35)
                source = driver.page_source
                def pdsuccess():
                    try:
                        finds=driver.find_element(by=By.CSS_SELECTOR,value='#folder0 > div.opened > div:nth-child(2)')
                        pd=True
                    except:
                        finds=None
                        pd=False
                    return pd
                
   
                for rs in range(5):
                    try:
                        pd=pdsuccess()
                        if 'Maximum number' in bs(source, features="lxml").text and pd==True:
                            # driver.close()
                            shuishui, shijianbuyu1234 = password[str(rs+1)]
                            piclink2 = f"https://{shuishui}:{shijianbuyu1234}@" + piclink.split('https://')[1]
                            driver.get(piclink2)
                            source = driver.page_source
                            print(piclink2)
                            driver.implicitly_wait(35)
                            pd=pdsuccess()
                            # print(driver.title)
                        else:
                            break
                    except:
                        pass
                print(piclink2)
                r += 1
                while True:
                    time.sleep(60)
                    if getcurrent_download_count(outdir)<2:#!如果下载文件数量小于3，等待60秒，则进行下一个循环
                        time.sleep(60)
                        break
        while True:
            time.sleep(60)
            if getcurrent_download_count(outdir)==0:#!如果下载文件数量等于0，跳出循环，等待60秒
                time.sleep(60)
                break
        delete_errorfile(outdir)#!删除未完成，同时删除下载多余的文件
        os.system("taskkill /im chrome.exe /f")
        
        
    def  downloadS3byIDM(self,outdir,df):
        outpicdirs=os.path.join(outdir,'File')
        if not os.path.exists(outpicdirs):
            os.makedirs(outpicdirs)
        
        call([IDM, '/s'])
        # outdir=r"Z:\WSR\GO2\Data/"
        wronglink=[]
        num=0
        piclinks=df['link']
        title=df['title']
        for link,name in zip(piclinks,title):
            shuishui='shuishui'
            shijianbuyu1234='shijianbuyu1234'
            link=f"https://{shuishui}:{shijianbuyu1234}@"+link.split('https://')[1]
            outfile=os.path.join(outpicdirs,name)+'.zip'
            if not os.path.exists(outfile):
                try:
                    call([IDM, '/d', link, '/p', outpicdirs, '/n', '/a'])
                    print('添加'+name)
                    # data=requests.get(link).content
                    # with open(outfile,'wb') as f:
                    #     f.write(data)
                    # if num%200:
                    #     time.sleep(60*60*2)#休息会
                    #     call([IDM, '/s'])
                except :
                    wronglink.append(link)
        # IDM=r"E:\idm绿色直装\ucbug.com-idm\idm 6.39.2.2\IDMan.exe"
        call([IDM, '/s'])
        
        call([IDM, '/s'])
        print('启动下载')
    # http://admin:Password@192.168.1.1
def delete_errorfile(outdir):
    """删除'.crdownload'文件以及多余的文件
    未下载完成文件以及下载多余的文件

    Args:
        outdir (_type_): _description_
    """

    files=glob.glob(outdir+'/*')
    for file in files:
        if os.path.splitext(file)[1]=='.crdownload':
            os.remove(file)
        if '(' in file and ')' in file:
            os.remove(file)
    
def changedownloaddir(outdir):
    prefs_file = os.path.join(r'C:\Users\SHUI\AppData\Local\Google\Chrome\User Data\Default', 'Preferences')

    with open(prefs_file, 'r', encoding='utf-8') as file:
        prefs_data = file.read()
    import json
    prefs_data=json.loads(prefs_data)
    prefs_data['savefile']['default_directory']=os.path.abspath(outdir)
    prefs_data['download']['default_directory']=os.path.abspath(outdir)
    # 在 Preferences 文件中替换下载路径
    # prefs_data = prefs_data.replace('"download.default_directory": "原始下载目录"', f'"download.default_directory": "{os.path.abspath(outdir)}"')

    with open(prefs_file, 'w', encoding='utf-8') as file:
        file.write(json.dumps(prefs_data))



now=datetime.datetime.now()
dates=now.strftime('%Y%m%d')
delta = datetime.timedelta(days=5)
cha=now-delta
startdate='20230706'
startdate=cha.strftime('%Y%m%d')
enddate=dates
geometry=r'K:\矢量数据\滇池\dianchi.shp'
inputdir='G:\Remote_Data_Cope\Romote Data\Sentinel-3'#!以修改这个为主


outdir=f'{inputdir}\File/'
T=downfiles(startdate,enddate)
df=T.downloadlinksS3( geometry,protype='OL_1_EFR___')
time.sleep(10)
T.downloadjpg(outdir,df)
T.downloadS3filebydriver(outdir,df)
# T.downloadS3byIDM(outdir,df)
time.sleep(5)
zipfiles=glob.glob(outdir+'/ZIP/*.zip')
putzipdir=outdir+'/unzipdata'
if not os.path.exists(putzipdir):
    os.makedirs(putzipdir)
cope_S2_from_GEE(putzipdir,'1').unzip(zipfiles)
print('解压完成')



# import psutil

# def get_chrome_process_count():
#     chrome_count = 0
#     for proc in psutil.process_iter(['name']):
#         if proc.info['name'] == 'chrome' or proc.info['name'] == 'chrome.exe':
#             chrome_count += 1
#     return chrome_count

# chrome_count = get_chrome_process_count()
# print(f"谷歌浏览器打开个数：{chrome_count}")




# # 打开Chrome浏览器的下载页面
# driver.get('chrome://downloads')
# page=driver.page_source
# cont=bs(page, features="lxml")
# cont.find_all(id='description')

# task_count = driver.execute_script("return document.querySelector('downloads-manager').shadowRoot.querySelectorAll('downloads-item[progress]').length")

# # 输出正在下载的任务个数
# print("当前正在下载的任务个数:", task_count)

def unzip(zipfiles,putzipdir):
    if not os.path.exists(putzipdir):
        os.mkdir(putzipdir)
    for file in zipfiles:
        try:
            if zipfile.is_zipfile(file):
                unzipoutdirl = os.path.join(putzipdir, os.path.split(file)[1].split('.')[0])
                z = zipfile.ZipFile(file, "r")
                outfile = os.path.join(unzipoutdirl, z.namelist()[0].split('/')[0])
                if not os.path.exists(unzipoutdirl):
                    os.mkdir(unzipoutdirl)
                if os.path.exists(outfile):
                    continue
                z.extractall(unzipoutdirl)
                z.close()
            else:
                continue
        except:
            pass  # 加不加都无所谓，但是加了可以跳过这个出现错误的代码块
        continue

