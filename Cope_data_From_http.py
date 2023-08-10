# encoding: utf-8
'''
 @author: shui
 @contact: 2232607787@qq.com
 @file: Cope_data_From_http.py
 @time: 2023/4/25 12:40
 @productname: langfeng
 @说明：
'''
import pandas as pd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
# from geo.Geoserver import Geoserver
# geo = Geoserver('http://localhost:8090/geoserver', username='admin', password='geoserver')

import numpy as np
from osgeo import ogr
import json
from sentinelsat import SentinelAPI
import os
import geopandas as gpd


	
class Copedata_fromhttp:
	def __init__(self):
		pass

# S2=downfiles('20230301','20230425')
# S2link=S2.downloadlinksS2(geometry="K:\工作\洪泽湖\layers\包含大多数河流\layers/POLYGON.shp",percent=24)
# # S2link.to_csv("K:\工作\洪泽湖/outdata.csv",encoding='gbk')

