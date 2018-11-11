# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:11:45 2018

@author: 严斌
"""

#决策树  处理非线性关系模型

import pandas as pd #用于数据分析
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer #特征转化器
from sklearn.tree import DecisionTreeClassifier #导入单一决策树分类器
from sklearn.ensemble import RandomForestClassifier #导入随机森林分类器
from sklearn.ensemble import GradientBoostingClassifier #导入梯度提升决策树分类器
from sklearn.metrics import classification_report #导入性能评估模块


titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt') #下载信息  

X = titanic[['pclass','age','sex']] #特征选择
y = titanic['survived'] #标记

X['age'].fillna(X['age'].mean(),inplace = True) #使用平均值对数据进行补全 inplace=True 修改原数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25 ,random_state = 33)

vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))#特征抽取，数值型保持不变，类别型单独抽离出来
#同样的测试集也要进行特征抽取
X_test = vec.transform(X_test.to_dict(orient = 'record'))#特征向量化处理

#使用分割后的训练数据进行模型学习

#单一决策树
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dct_y_predict = dtc.predict(X_test)

print("使用决策树模型准确率：",dtc.score(X_test,y_test))
print(classification_report(dct_y_predict,y_test,target_names=["死掉","幸存"]))

#随机森林
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)

print("使用随机森林模型准确率：",rfc.score(X_test,y_test))
print(classification_report(rfc_y_predict,y_test,target_names=["死掉","幸存"]))

#梯度提升决策树
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict = gbc.predict(X_test)

print("使用梯度提升决策树模型准确率：",gbc.score(X_test,y_test))
print(classification_report(gbc_y_predict,y_test,target_names=["死掉","幸存"]))



