import random
import string
import pickle
import numpy as np
import pandas as pd
import tables
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import fastparquet as fp

class TableFormat:
    def __init__(self, format):
        self.format = format
        
    def write(self, data, filename):
        raise NotImplementedError('Subclasses must implement this method')
    
    def read(self, filename):
        raise NotImplementedError('Subclasses must implement this method')


class PickleFormat(TableFormat):
    def write(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def read(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class NumpyFormat(TableFormat):
    def write(self, data, filename):
        np.save(filename, data)
    
    def read(self, filename):
        return np.load(filename)


class PandasCSVFormat(TableFormat):
    def write(self, data, filename):
        data.to_csv(filename, index=False)
    
    def read(self, filename):
        return pd.read_csv(filename)


class PandasExcelFormat(TableFormat):
    def write(self, data, filename):
        data.to_excel(filename, index=False)
    
    def read(self, filename):
        return pd.read_excel(filename)


class HDF5Format(TableFormat):
    def write(self, data, filename):
        with tables.open_file(filename, mode='w') as file:
            file.create_array('/', 'data', obj=data)
    
    def read(self, filename):
        with tables.open_file(filename, mode='r') as file:
            return file.root.data[:]


class PyArrowFeatherFormat(TableFormat):
    def write(self, data, filename):
        table = pa.Table.from_pandas(data)
        feather.write_feather(table, filename)
    
    def read(self, filename):
        table = feather.read_feather(filename)
        return table.to_pandas()


class PyArrowParquetFormat(TableFormat):
    def write(self, data, filename):
        table = pa.Table.from_pandas(data)
        pq.write_table(table, filename)
    
    def read(self, filename):
        table = pq.read_table(filename)
        return table.to_pandas()


class FastParquetFormat(TableFormat):
    def write(self, data, filename):
        table = pa.Table.from_pandas(data)
        fp.write(filename, table)
    
    def read(self, filename):
        table = fp.ParquetFile(filename).to_pandas()
        return table


def write_table(data, filename, format):
    formats = {
        'pickle': PickleFormat,
        'numpy': NumpyFormat,
        'pandas_csv': PandasCSVFormat,
        'pandas_excel': PandasExcelFormat,
        'hdf5': HDF5Format,
        'pyarrow_feather': PyArrowFeatherFormat,
        'pyarrow_parquet': PyArrowParquetFormat,
        'fastparquet': FastParquetFormat
    }
    if format not in formats:
        raise ValueError(f'Unsupported format: {format}')
    table_format = formats[format]()
    table_format.write(data, filename)


def read_table(filename, format):
    formats = {
        'pickle': PickleFormat,
        'numpy': NumpyFormat,
        'pandas_csv': PandasCSVFormat,
        'pandas_excel': PandasExcelFormat,
        'hdf5': HDF5Format,
        'pyarrow_feather': PyArrowFeatherFormat,
        'pyarrow_parquet': PyArrowParquetFormat,
        'fastparquet': FastParquetFormat
    }
    if format not in formats:
        raise ValueError(f'Unsupported format: {format}')
    table_format = formats[format]()
    return table_format.read(filename)
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet as fp

def append_to_pyarrow_table(new_data, filename):
    # 将新的数据转换为一个PyArrow表
    new_table = pa.Table.from_pandas(new_data)
    
    # 从文件中读取原始表格
    original_table = pq.read_table(filename)
    
    # 使用pa.concat_tables()函数将原始表格和新表格连接起来
    merged_table = pa.concat_tables([original_table, new_table])
    
    # 检查是否有重复的行，并删除它们
    unique_table = merged_table.drop_duplicates()
    
    # 如果新数据已经全部存在于原始表格中，则不进行任何操作
    if len(unique_table) == len(original_table):
        return
    
    # 将处理后的表格写入文件
    pq.write_table(unique_table, filename)

def append_to_fastparquet_table(new_data, filename):
    # 将新的数据转换为一个Pandas数据框
    new_df = pd.DataFrame(new_data)
    
    # 从文件中读取原始数据框
    original_df = fp.ParquetFile(filename).to_pandas()
    
    # 连接原始数据和新数据
    merged_df = pd.concat([original_df, new_df], ignore_index=True)
    
    # 删除重复的行
    unique_df = merged_df.drop_duplicates()
    
    # 如果新数据已经全部存在于原始数据框中，则不进行任何操作
    if len(unique_df) == len(original_df):
        return
    
    # 将处理后的数据框写入文件
    fp.write(filename, unique_df)
