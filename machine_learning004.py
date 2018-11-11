# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:40:49 2018

@author: 严斌
"""

#线性回归器
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR #支持向量机回归模块
from sklearn.neighbors import KNeighborsRegressor #K近邻回归器
from sklearn.tree import DecisionTreeRegressor #单一回归树模型
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

from sklearn.linear_model import LinearRegression#此处使用两种线性回归器比较
from sklearn.linear_model import SGDRegressor

boston = load_boston()#载入波士顿房价信息

X = boston.data  #划分数据集
y = boston.target #目标
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 33)


##特征和目标值标准化
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))


#训练模型
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)
print("解析方法的评估：",lr.score(X_test,y_test))

sgdr = SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict = sgdr.predict(X_test)
print("随机梯度法的评估：",sgdr.score(X_test,y_test))

linear_svr = SVR(kernel='linear') #使用线性核函数的支持向量机
linear_svr.fit(X_train,y_train)
linear_y_predict = linear_svr.predict(X_test)
print("线性核函数性能评估：",linear_svr.score(X_test,y_test))

poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_y_predict = poly_svr.predict(X_test)
print("多项式核函数性能评估：",poly_svr.score(X_test,y_test))

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_y_predict = rbf_svr.predict(X_test)
print("径向基核函数性能评估：",rbf_svr.score(X_test,y_test))

knr = KNeighborsRegressor()
knr.fit(X_train,y_train)
knr_y_predict = knr.predict(X_train)
print("K近邻性能评估：",knr.score(X_test,y_test))

uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_y_predict = uni_knr.predict(X_test)
print("K近邻(平均回归)性能评估:",uni_knr.score(X_test,y_test))

dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_y_predict = dis_knr.predict(X_test)
print("K近邻（距离加权回归）性能评估：",dis_knr.score(X_test,y_test))

dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict = dtr.predict(X_test)
print("单一回归树性能评估：",dtr.score(X_test,y_test))

rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr = rfr.predict(X_test)
#print("随机森林性能评估：",rfr.score(X_test,y_test))

etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict = etr.predict(X_test)
print("极端随机森林性能评估：",etr.score(X_test,y_test))

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict = gbr.predict(X_test)
print("梯度提升性能评估：",gbr.score(X_test,y_test))


