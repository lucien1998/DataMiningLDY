import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
import math
from mlxtend.preprocessing import TransactionEncoder as TE
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules as ar

def deal(data):
    return data.dropna().tolist()
df_arr = df.apply(deal,axis=1).tolist() # 转化成列表
te = TE()  # 定义模型
df_tf = te.fit_transform(df_arr)
df = pd.DataFrame(df_tf,columns=te.columns_)
df
freq_itemsets = apriori(df,min_support=0.005,use_colnames=True)
freq_itemsets.sort_values(by='support',ascending=False,inplace=True)
freq_itemsets
a_r = ar(freq_itemsets,metric='lift')
a_r = a_r.sort_values(by='lift',ascending=False).reset_index(drop=True)
a_r
t = []
for i in range(a_r.shape[0]):
    item = a_r.iloc[i]
    t.append(item.support/math.sqrt(item['antecedent support']*item['consequent support']))
a_r['cosine'] = t
t = []
for i in range(a_r.shape[0]):
    item = a_r.iloc[i]
    t.append(0.5*(item.support/item['antecedent support']+item.support/item['consequent support']))
a_r['Kulc'] = t
t = []
for i in range(a_r.shape[0]):
    item = a_r.iloc[i]
    t.append(item.support/(item['antecedent support']+item['consequent support']-item.support))
a_r['Jaccard'] = t
a_r = a_r.sort_values(by='lift',ascending=False).reset_index(drop=True)
a_r.head(30)

plt.xlabel('support')
plt.ylabel('confidence')
for i in range(a_r.shape[0]):
    plt.scatter(a_r.support[i],a_r.confidence[i],s=20,c='b',alpha=(a_r.lift.iloc[i])/(a_r.lift.iloc[0])*0.8/(a_r.lift.iloc[0]-a_r.lift.iloc[-1])+0.2)
plt.xlabel('support')
plt.ylabel('confidence')
a_r = a_r.sort_values(by='cosine',ascending=False).reset_index(drop=True)
for i in range(a_r.shape[0]):
    plt.scatter(a_r.support[i],a_r.confidence[i],s=20,c='b',alpha=(a_r.cosine.iloc[i])/(a_r.cosine.iloc[0]))
plt.xlabel('support')
plt.ylabel('confidence')
a_r = a_r.sort_values(by='Kulc',ascending=False).reset_index(drop=True)
for i in range(a_r.shape[0]):
    plt.scatter(a_r.support[i],a_r.confidence[i],s=20,c='b',alpha=(a_r.Kulc.iloc[i])/(a_r.Kulc.iloc[0]))
plt.xlabel('support')
plt.ylabel('confidence')
a_r = a_r.sort_values(by='Jaccard',ascending=False).reset_index(drop=True)
for i in range(a_r.shape[0]):
    plt.scatter(a_r.support[i],a_r.confidence[i],s=20,c='b',alpha=(a_r.Jaccard.iloc[i])/(a_r.Jaccard.iloc[0]))
