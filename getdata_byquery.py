# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: getdata_byquery.py
 @time: 2023/6/5 11:19
 @productname: Code
 @说明：
'''
import pandas as pd

import os
import glob


class WaterQualityData:
    need = {
        '云南省': ['洱海湖心', '滇池南', '草海中心']
    }
    def __init__(self,allneeddir) -> None:
        self.allneeddir=allneeddir
        self.alltxt=glob.glob(os.path.join(self.allneeddir,'*', '*.txt'))
        self.allsates=[os.path.basename(os.path.splitext(file)[0]).split('-')[1] for file in self.alltxt]

    def get_need_state(self,needstates=None,allstate=False):
        """根据need中的站点查询需要的水质数据

        allneed:path of waterqulity,like:K:\实测数据下载\auto_waterquality_download\waterquality
        """
        if needstates is None:
            dirs = os.listdir(self.allneeddir)
            needstatedata = {}
            for dir in dirs:
                for pro in self.need.keys():
                    if dir == pro:
                        statefiles = glob.glob(os.path.join(self.allneeddir, dir, '*.txt'))
                        needstate = self.need[pro]
                        ns = needstatedata.setdefault(pro, [])
                        for st in needstate:
                            for hasst in statefiles:
                                if st in hasst:
                                    ns.append(hasst)
            return needstatedata
        else:
            dirs = os.listdir(self.allneeddir)
            needstatedata = {}
            for dir in dirs:
                statefiles = glob.glob(os.path.join(self.allneeddir, dir, '*.txt'))
                # needstate = self.need[pro]
                if allstate==True:
                    needstates=self.allsates

                for st in needstates:
                    for hasst in statefiles:
                        import regex as re
                        # pattern = r"[-.](.*?)[-.]"
                        pattern=r'-(.*)\.'
                        namesmatches = re.findall(pattern, hasst)
                        if st == namesmatches[0]:
                            ns = needstatedata.setdefault(dir, [])
                            ns.append(hasst)
            return needstatedata

    def getcsvdata(self,file):
        liuyu,zhandian=os.path.splitext(os.path.basename(file))[0].split('-')
        col = ['流域','站点','发布时间', '水质类别', '水温', 'PH值', '溶解氧', '电导率', '浊度', '高锰酸盐指数', '氨氮', '总磷', '总氮', '叶绿素', '藻密度','站点情况']
        data = pd.read_csv(file, encoding='utf-8', header=None, delimiter='\t').iloc[:,:14]
        liuyus=[liuyu]*data.shape[0]
        zhandians=[zhandian]*data.shape[0]
        data.insert(loc=0,column='流域',value=liuyus)
        data.insert(loc=1,column='站点',value=zhandians)
        data.columns = col
        newtime=pd.Series(data=pd.DatetimeIndex(data['发布时间']).strftime('%Y/%m/%d %H:%M'),name='发布时间')
        data.update(newtime)
        # data['发布时间'].update()
        return data

    def select_data(self, need_pro, need_freq, need_date):
        """
        根据指定的站点、频率和日期，选择对应的水质数据

        Parameters:
        ----------
        need_pro : list of str
            需要查询的产品
        need_freq : str
            需要查询的数据频率
        need_date : list of str
            需要查询的日期
        all_need : str
            水质数据文件路径

        Returns:
        -------
        need_data : dict of pd.DataFrame
            选中的水质数据
        """
        needstatedata = self.get_need_state()
        need_data = {}
        for ke in needstatedata.keys():
            files = needstatedata[ke]
            for file in files:
                # file=self.alltxt[1]
                try:
                    data=self.getcsvdata(file)
                    name = ke + '_' + os.path.basename(os.path.splitext(file)[0])
                    date = pd.to_datetime(data['监测时间'])
                    index = []
                    for dat in need_date:
                        for r, hasda in enumerate(list(date.dt.strftime('%Y%m%d'))):
                            if dat == hasda:
                                index.append(r)
                    data.index = pd.PeriodIndex(data=date, freq=need_freq)
                    needata = data.loc['2023-04'][['监测时间'] + need_pro].iloc[index, :]
                    need_data[name] = needata
                except:
                    pass
        return need_data
    def exportdata(self,needstates=None):
        if needstates is None:
            needstates=self.allsates
        needstatedata = self.get_need_state(needstates=needstates)
        need_data = []
        for ke in needstatedata.keys():
            files = needstatedata[ke]
            for file in files:
                data=self.getcsvdata(file)
                need_data.append(data)
        alldata=pd.concat(need_data)
        return alldata


# WQ=WaterQualityData(r"K:\BaiduNetdiskDownload\waterquality/")

# data=WQ.exportdata(['浦阳江出口','闸口'])

# index=[r for r,s in enumerate(list(data['发布时间'])) if '00:00' in s]
# index=list(range(len(data['发布时间'])))

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# plt.figure(figsize=(10,5))
# ds=data['总磷'].iloc[index]
# plt.scatter(x=data['发布时间'].iloc[index], y=ds)

# # 创建一个新的MultipleLocator对象，将横坐标刻度分隔为500
# x_major_locator = ticker.MultipleLocator(50)

# # 获取当前的坐标轴，并设置刻度定位器
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

# # 倾斜横坐标刻度标签
# plt.xticks(rotation=45)
# plt.tight_layout()
# # 显示图形
# plt.show()
# # len(glob.glob("K:\BaiduNetdiskDownload\waterquality/*/*.txt"))

# #!!
# cors=data.corr()

# plt.imshow(cors,'jet')
# plt.colorbar()
# plt.show()

# import seaborn as sns
# import numpy as np

# # sns.heatmap(cors)

# ax=sns.heatmap(data=cors, cmap='plasma', annot=True, linewidth=0.9, linecolor='white',
#             cbar=True,  vmin=0, center=0, square=False,
#             robust=True,
#             annot_kws={'color': 'white', 'size': 10, 'family':'Simsun', 'style': None, 'weight': 10},
#             cbar_kws={'orientation': 'vertical', 'shrink': 1, 'extend': 'max', })
# # 设置X轴刻度字体
# # ax.xaxis.set_tick_params(labelsize=18, font='SimSun') 
# ax.xaxis.set_ticklabels( list(cors.keys()),font='SimSun')
# ax.yaxis.set_ticklabels( list(cors.keys()),font='SimSun')
# # 设置Y轴刻度字体 
# # ax.yaxis.set_tick_params(labelsize=18, font='SimSun')
# plt.tick_params(axis='x',labelsize=18,rotation=90)
# plt.tick_params(axis='y',labelsize=18,rotation=0)
# # plt.xticks(rotation=90)
# # plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()

# plt.scatter(x=data['水温'],y=data['溶解氧'])
# plt.show()



# np.corrcoef(data['水温'],data['总氮'])