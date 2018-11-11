# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:57:18 2018

@author: 严斌
"""

#手写体数字识别
from sklearn.datasets import load_digits #导入数据集
from sklearn.cross_validation import train_test_split #数据分割
from sklearn.preprocessing import StandardScaler #数据标准化模块
from sklearn.svm import LinearSVC #从sklearn.svm中导入线性假设的支持向量机分类器LinearSVC

digits = load_digits()#加载图像数据
#print(digits.data.shape)#查看数据规模和特征维度  1796 64
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.25,random_state = 33)
#数据集，标签，训练集的比例，设置随机数种子
#X_train和X_test为训练题和考试题
#y_train和y_test为标准答案

#对需要训练和测试的特征数据进行标准化
ss = StandardScaler()##标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导。
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC() #初始化分类器
lsvc.fit(X_train,y_train) #训练模型
y_predict = lsvc.predict(X_test[[9]]) #对测试集进行预测
print(y_predict,y_test[[9]])
print("该模型的准确率为:",lsvc.score(X_test,y_test))

