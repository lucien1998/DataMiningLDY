import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV


name_data = df[['Name','Global_Sales','Year','Genre']].sort_values(by='Global_Sales', ascending=False).head(10)
name_data

plt.figure(figsize=(20, 10))
sns.barplot(y='Global_Sales',x='Name',data=name_data)
plt.xticks(plt.xticks()[0], rotation=90, size=16)
plt.tight_layout()
plt.xlabel('Name', fontsize = 20)
plt.ylabel('Global_Sales', fontsize = 20)
plt.show()

genre_data = df.groupby(['Genre']).sum().loc[:,'Global_Sales'].sort_values(ascending=False)
plt.figure(figsize=(20, 6))
sns.barplot(y=genre_data.values, x=genre_data.index)
plt.xlabel('Genres')
plt.ylabel('Global_Sales')
plt.title('Global_Sales of Genres')
plt.show()

platform_data = df.groupby(['Platform']).count().loc[:,'Name'].sort_values(ascending=False)
plt.figure(figsize=(20,12))
sns.barplot(y=platform_data.values, x=platform_data.index)
plt.xlabel('Platform')
plt.ylabel('counts')
plt.show()

publisher = df.groupby(['Publisher']).sum().loc[:,'Global_Sales'].sort_values(ascending=False)
publisher_data = publisher[publisher.values > 100]
plt.figure(figsize=(20,16))
sns.barplot(y=publisher_data.values, x=publisher_data.index)
plt.xticks(plt.xticks()[0], rotation=90, size=16)
plt.tight_layout()
plt.title('Global_Sales of Publisher', fontsize = 20)
plt.xlabel('Publisher', fontsize = 20)
plt.ylabel('Global_Sales', fontsize = 20)
plt.show()

df.groupby('Year').count()
df = df[df.Year <= 2016]
y = df.Global_Sales
x = df.drop(['Name','Global_Sales'], axis=1)
LE = LabelEncoder()
x.Platform = LE.fit_transform(df.Platform)
x.Genre = LE.fit_transform(df.Genre)
x.Publisher = LE.fit_transform(df.Publisher)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

#使用交叉验证进行参数调优
Tree = DecisionTreeRegressor(random_state=2)
param_grid = {
     'max_depth': [10, 15],
     #'min_sample_leaf': [30, 50, 100],
     'splitter': ('random', 'best')
}
grid_search = GridSearchCV(Tree, param_grid, scoring='r2', cv=10)
grid_search.fit(X_train, y_train)
print('训练集best_score_: ' + str(grid_search.best_score_))
print('测试集score: ' + str(grid_search.score(X_test, y_test)))  # 分数
print(grid_search.best_params_)  # 最好的超参数
print(grid_search.best_estimator_) #决策树的参数设置

#使用调好的参数训练并预测
model = grid_search.best_estimator_.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print('mean_squared_error:', mean_squared_error(y_test, y_pred))
print('r2_score:', r2_score(y_test, y_pred))


plt.rcParams['font.sans-serif']=['STSong']
plt.rcParams['axes.unicode_minus'] = False
sales_table = df.pivot_table(index='Name',values=['NA_Sales','EU_Sales','JP_Sales','Global_Sales'],aggfunc=np.sum)
property_NA = sales_table['NA_Sales'].sum()/sales_table['Global_Sales'].sum()
property_EU = sales_table['EU_Sales'].sum()/sales_table['Global_Sales'].sum()
property_JP = 1 - property_EU - property_NA
#property_JP = sales_table['JP_Sales'].sum()/sales_table['Global_Sales'].sum()
properties = (property_NA,property_EU,property_JP)
property_label = ('NA_Sales','EU_Sales','JP_sales')
print(property_NA,property_EU,property_JP)
plt.figure(figsize = (6,6))
plt.title("不同地区的游戏销量分布", fontsize=18)
plt.pie(x=properties, labels=property_label, autopct="%0.1f%%", shadow=False)
plt.show()

Genre_NA = df.groupby(['Genre']).sum().loc[:,'NA_Sales'].sort_values(ascending=False)
Genre_EU = df.groupby(['Genre']).sum().loc[:,'EU_Sales'].sort_values(ascending=False)
Genre_JP = df.groupby(['Genre']).sum().loc[:,'JP_Sales'].sort_values(ascending=False)
Genre_NA = Genre_NA[Genre_NA.values > 60]
Genre_EU = Genre_EU[Genre_EU.values > 60]
Genre_JP = Genre_JP[Genre_JP.values > 60]
label_NA = Genre_NA._stat_axis.values.tolist()
label_EU = Genre_EU._stat_axis.values.tolist()
label_JP = Genre_JP._stat_axis.values.tolist()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize = (6,6))
plt.title("北美地区的游戏类型占比", fontsize=18)
plt.pie(x = Genre_NA,labels = label_NA,autopct = "%0.1f%%",shadow = True)
plt.show()
plt.figure(figsize = (6,6))
plt.title("欧洲地区的游戏类型占比", fontsize=18)
plt.pie(x = Genre_EU,labels = label_EU,autopct = "%0.1f%%",shadow = True)
plt.show()
plt.figure(figsize = (6,6))
plt.title("日本地区的游戏类型占比", fontsize=18)
plt.pie(x = Genre_JP,labels = label_JP,autopct = "%0.1f%%",shadow = True)
plt.show()


genre_tend = df.groupby(['Genre'])['Global_Sales'].sum().sort_values(ascending=False).head(4).index
genre_tend = df[df.Genre.isin(genre_tend)]
plt.figure(figsize = (10,6))
sns.lineplot(x='Year', y='Global_Sales', hue='Genre', data=genre_tend, ci=None)
plt.legend(loc='upper right')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Global_Sales', fontsize=14)
plt.title('四种最受欢迎的游戏类型的销售趋势', fontsize=16)
plt.show()

