# -*- coding: utf-8 -*-
"""
Created on Thu May 31 08:28:36 2018

@author: 严斌
"""

#朴素贝叶斯模型
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #导入用于文本特征向量转化的模块
from sklearn.naive_bayes import MultinomialNB #导入贝叶斯模型

from sklearn.metrics import classification_report #用于详细性能报告

#数据下载
news = fetch_20newsgroups(subset='all') #此时需要从网上即时下载数据
#print(len(news.data))
#print(news.data[0])

#数据分割
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size = 0.25 , random_state = 33)

#数据预处理
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#模型训练
mnb = MultinomialNB() #使用默认配置初始化模型
mnb = mnb.fit(X_train,y_train) #此处参数为训练集
y_predict = mnb.predict(X_test)

#模型性能评估
print("贝叶斯模型的准确率为:",mnb.score(X_test,y_test)) #此模型不适用与数据间特征关联性较强的分类任务
print(classification_report(y_test,y_predict,target_names = news.target_names))



