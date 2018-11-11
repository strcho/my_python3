# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:34:44 2018

@author: 严斌
"""

#无监督学习模型
#数据聚类

#K均值算法
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header = None)
#digits_train.to_pickle('datas/optdigits.csv')
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header = None)
  
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pre = kmeans.predict(X_test)
print(metrics.adjusted_rand_score(y_test,y_pre))



