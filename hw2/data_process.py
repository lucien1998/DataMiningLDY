import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
import youtube_process
from mlxtend.preprocessing import TransactionEncoder
file_US = "USvideos.csv"
US_data = pd.read_csv(file_US, keep_default_na=False, low_memory=False)
US_data
df = US_data[['category_id','views']]
df
with open("US_category_id.json", 'r') as f:
    content = json.load(f)
category_map = {}
for i in content['items']:
    category_map[int(i['id'])] = i['snippet']['title']
category_map
t = df['category_id'].map(category_map)
df = pd.concat([df,t],axis=1)
df.columns=['category_id','views','category']
grade = []
for i in df['views'].values:
    views_map = lambda x:{x>=4194399:'A',1823157<=x<4194399:'B',681861<=x<1823157:'C',
                          242329<=x<681861:'D',549<=x<242329:'E'}
    grade.append(views_map(i)[True])
df['views_grade'] = grade
df = df.drop(['category_id', 'views'], axis = 1)
df
def deal(data):
    return data.dropna().tolist()
df_arr = df.apply(deal,axis=1).tolist() # 转化成列表
TE = TransactionEncoder()  # 定义模型
df_tf = TE.fit_transform(df_arr)
df = pd.DataFrame(df_tf,columns=TE.columns_)
df