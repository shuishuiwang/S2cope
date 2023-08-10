# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: Vec_cope.py
 @time: 2023/5/26 11:43
 @productname: 空间舒适度分析
 @说明：
'''

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point,mapping
import os
from esda.moran import Moran_Local
import esda
import pandas as pd
from geopandas import GeoDataFrame
import libpysal as lps
import numpy as np
from shapely.geometry import Point
# import contextily as ctx
from pylab import figure, scatter, show
from splot.esda import plot_moran

class MoranAnalysis:
    def __init__(self, filename):

        '''':示例
        filename=r'K:\实测数据下载\全国水质爬取\全国水质数据\allstaticshparcgiscope_out.shp'
        data = MoranAnalysis(filename=filename)
        readdata=data.read_data()#先读取
        readdata.generate_weights().plot_neighbor_graph()#然后获取权重
        readdata.calculate_localmoran('经度').plot_local_moran()#然后绘制局部
        readdata.calculate_moran('经度').plot_moran_results()#打印全局
        '''
        self.filename = filename
        self.gdf = None
        self.wq = None
        self.wr = None
        self.centroids = None
        self.y = None
        self.mi = None
        self.plot_size = (8, 8)
    def read_data(self):
        self.gdf = gpd.read_file(self.filename)
        return self

    def print_columns(self):
        if self.gdf is not None:
            self.columns=self.gdf.columns.values
            print(self.columns)
            return self
        else:
            print("Data not read yet.")

    def plot_data(self, column_name):
        if self.gdf is not None:
            ax = self.gdf.plot(figsize=self.plot_size, column=column_name)
            ax.set_axis_off()
            plt.show()
        else:
            print("Data not read yet.")

    def generate_weights(self, weight_type='queen'):
        if self.gdf is not None:
            if weight_type == 'queen':
                self.wq = lps.weights.Queen.from_dataframe(self.gdf)
                self.wq.transform = 'r'  # 标准化矩阵
                self.centroids = self.gdf.geometry.centroid  # 计算多边形几何中心
            elif weight_type == 'rook':
                self.wr = lps.weights.Rook.from_dataframe(self.gdf)
                # wr.transform = 'r' # 标准化矩阵
                self.centroids = self.gdf.geometry.centroid  # 计算多边形几何中心
            else:
                print("Invalid weight type argument.")
            return self
        else:
            print("Data not read yet.")

    def plot_neighbor_graph(self, weights_type="queen"):
        # self.wr, self.centroids=self. generate_weights( weight_type=weights_type)
        if self.centroids is not None:
            if weights_type == "queen" and self.wq is not None:
                fig = figure(figsize=self.plot_size)
                plt.plot(self.centroids.x, self.centroids.y, '.')
                for k, neighs in self.wq.neighbors.items():
                    origin = self.centroids[k]
                    for neigh in neighs:
                        segment = self.centroids[[k, neigh]]
                        plt.plot(segment.x, segment.y, '-')
                plt.title('Queen Neighbor Graph')
                plt.axis('off')
                plt.show()
            elif weights_type == "rook" and self.wr is not None:
                fig = figure(figsize=self.plot_size)
                plt.plot(self.centroids.x, self.centroids.y, '.')
                for k, neighs in self.wr.neighbors.items():
                    origin = self.centroids[k]
                    for neigh in neighs:
                        segment = self.centroids[[k, neigh]]
                        plt.plot(segment.x, segment.y, '-')
                plt.title('Rook Neighbor Graph')
                plt.axis('off')
                plt.show()
            else:
                print("Invalid weights type or weight matrix not generated yet.")
        else:
            print("Weight matrix not generated yet.")

    def calculate_moran(self, column_name):
        if self.gdf is not None:
            self.y = np.float_(self.gdf[column_name].to_numpy())
            self.mi = esda.moran.Moran(self.y, self.wq)
            return self
        else:
            print("Data not read yet.")

    def calculate_localmoran(self, column_name):
        try:
            if self.gdf is not None and self.wq is not None:
                y = np.float_(self.gdf[column_name].to_numpy())
                self.moran_loc = Moran_Local(y=y, w=self.wq)
                return self
            else:
                raise ValueError("Data not read yet or wq not cope yet.")
        except Exception as e:
            print(f"Error calculating local Moran's I: {e}")

    def print_moran_results(self):
        # self.mi=self.calculate_moran( column_name)
        if self.mi is not None:
            print("Moran's I 值为：", self.mi.I)
            print("随机分布假设下Z检验值为：", self.mi.z_rand)
            print("随机分布假设下Z检验的P值为：", self.mi.p_rand)
            print("正态分布假设下Z检验值为：", self.mi.z_norm)
            print("正态分布假设下Z检验的P值为：", self.mi.p_norm)
        else:
            print("Moran analysis not performed yet.")

    def plot_moran_results(self):

        if self.mi is not None:

            plot_moran(self.mi, zstandard=True)
            plt.show()
        else:
            print("Moran analysis not performed yet.")
    def plot_local_moran(self,cmap='coolwarm'):

        # 绘制局部莫兰指数图
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ad = self.gdf.assign(cl=self.moran_loc.Is).plot(
            column='cl', categorical=True,
            k=2, cmap=cmap, alpha=0.8,
            linewidth=0.1, ax=ax,
            legend=False,
            legend_kwds={'loc': 'upper left', 'title': 'Significance'}
        )
        ax.set_title('Local Moran Scatterplot')
        ax.set_axis_off()
        # Add colorbar
        norm=plt.Normalize(vmin=0,  vmax=np.max(self.moran_loc.Is))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, orientation='horizontal', pad=0.05)
        cbar.set_label('Cluster Significance')
        plt.show()

# file=r"K:\矢量数据\china-latest-free.shp\gis_osm_traffic_free_1.shp"

class Vec_cope:
    def __init__(self, shpfile):
        self.shpfile = shpfile

    def show(self):
        gdf = gpd.read_file(self.shpfile)
        gdf.plot(cmap='jet')
        plt.title(os.path.basename(self.shpfile))
        plt.show()
    def ceateshp(self):
         pass

    def getallgeoms(self, gdf, intervel=100):
        alldata = []
        n = 0
        for s in range(len(gdf.geometry)):
            geomes = gdf.geometry[s]
            allpoint = mapping(geomes)
            for r in range(len(allpoint['coordinates'][0])):
                n += 1
                if n % intervel == 0:
                    # print(allpoint['coordinates'][0][r])
                    point = allpoint['coordinates'][0][r]
                    geom = Point(point)
                    geodf = gpd.GeoSeries(geom, crs=4326).to_crs('epsg:3326').buffer(400)
                    alldata.append(geodf)
        return alldata

    def Dengju_sampling(self, roads, dist=5):
        # roadslength = roads.to_crs(epsg=3326)
        
        
        lines = roads.to_crs('epsg:32619').geometry.apply(lambda x: LineString(x))
        # 分割路网，并对每个分割后的线段进行采样
        
        points = []
        for i in range(len(lines)):
            # print(i)
            try:
                line = lines[i]
            except:
                if len(lines)==1:
                    line=lines
            
            length = line.length
            # lengt+=length
            # print()
            num_samples = int(length)
            # 对线段进行采样
            # dist=50
            for j in range(1, num_samples,dist):
                
                point = line.interpolate(j)
                try:
                    points.append(point.iloc[0])
                except:
                    points.append(point)
        data = gpd.GeoSeries(points, crs='epsg:32619',name='outpoint').to_crs('epsg:4326')
        lon = [s.x for s in data]
        lat = [s.y for s in data]
        need = {
            'lon':lon,
            'lat':lat,
        }

        Gdf = gpd.GeoDataFrame(data=need, geometry=list(data.geometry),crs='epsg:4326')
            # data.plot('jet')
            # plt.show()
        return Gdf


