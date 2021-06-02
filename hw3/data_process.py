import pandas as pd
import numpy as np

#读取数据
df = pd.read_csv("vgsales.csv")

#查看数据
df.head()
df.info()

#查看缺失值
df.isnull().sum()

#删除缺失值
df.dropna(inplace=True)

#查看处理缺失值后的数据
df.isnull().sum()
df.describe()