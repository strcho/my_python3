# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:18:52 2018

@author: 严斌
"""

#k 近邻模型  此模型属于无参数模型，具有很高的计算复杂度和内存消耗 数据规模巨大时计算时间代价贼高！！

from sklearn.datasets import load_iris # 用于导入数据集 
from sklearn.cross_validation import train_test_split #数据分割
from sklearn.preprocessing import StandardScaler #用于数据预处理
from sklearn.neighbors import KNeighborsClassifier #导入k近邻分类器
from sklearn.metrics import classification_report #用于结果评估


iris = load_iris()
#print(iris.DESCR)#查看数据说明

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.25 , random_state = 33)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test) #此处使用fit_transform会降低模型准确率！！！

knc = KNeighborsClassifier()
knc.fit(X_train,y_train)

y_predict = knc.predict(X_test)

print("k近邻模型的准确率为:",knc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=iris.target_names))




