#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:43:17 2020

@author:Vamsi Abbireddy
"""
# importing all the required modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
a = pd.read_csv("/Users/egreddy/Downloads/train.csv.rimzydo (1).partial")
#looking at the data
a.head()
a.columns
#copying the data into a duplicate dataframe
X = a.copy()

#looking for the null values if any
X.isna().sum()


#Taking out the target variable from the dataset 
y = a["SalePrice"]
del X["SalePrice"]


#Filling the  null values of the continous columns with mean
X["LotFrontage"].fillna(X["LotFrontage"].mean(), inplace = True)
X["MasVnrArea"].fillna(X["MasVnrArea"].mean(), inplace = True)
X.isna().sum()

for i in X:
    if X[i].isna().any():
        X[i].fillna(X[i].mode()[0], inplace = True)
        #ss = pd.get_dummies(X[i])
        #X = pd.concat([ss, X])
		#del X[i]
		
X.isna().sum()
		
		
		
		
		
#dealing with the correlation

corr = X.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False

#Taking all the needed columns into a dataset               
X = X[["Id", "MSSubClass",  "LotFrontage", "LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF",
	   "LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageYrBlt","GarageCars","GarageArea","WoodDeckSF",
	   "OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold"]]

X.drop(X.columns[[0,13,23,25,27]], axis = 1, inplace = True)

#figuring out the variables which have outliers using boxplots. 
#plt.boxplot(X.iloc[:,0])
#plt.boxplot(X.iloc[:,1])

#capping the variables which have outliers
per = [2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,22,23,24,25,26,27,28,29,30,31]
for i in per :
	percentiles = X.iloc[:,i].quantile([0.005,0.97]).values
	X.iloc[:,i] =  X.iloc[:,i].clip(percentiles[0], percentiles[1])



#Calculating the cost function
def cost_function(pred, y):
	m = X.shape[0]
	J = 1/(2*m) * sum((pred - y.T)**2)
	
	return J

#Adding the extra consant column 
def addconstant(X):
	ones = pd.DataFrame(np.ones([X.shape[0],1]), index = X.index)
	X.insert(0, "Constant", ones)
	
	return X

"""Gradient descent algorithm : In this algorithm it has
	normalising feature builted in it along with the converging algorithm."""
def gradient_descent(X, y, LR, iterations):
	ones = pd.DataFrame(np.ones([X.shape[0],1]), index = X.index)
	X = ((X - np.mean(X.values))/np.std(X.values))
	#X = pd.concat([ones,pd.DataFrame(X)], axis = 1)
	X.insert(0, "Constant", ones)
	X = ((X - np.mean(X.values))/np.std(X.values))	
	theta = pd.DataFrame(np.zeros([X.shape[1],1]))
	h_X = hypothesis(X, theta)
	m = X.shape[0]
	y = np.array([y.values])
	for i in range(iterations):
		h_X =hypothesis(X, theta)
		theta =theta - pd.DataFrame((LR/m) * ((np.dot(X.T, (h_X - y.T))).reshape(theta.shape[0], 1)))
		
	return theta, h_X, cost_function(h_X, y)

# Function for predicting the values with the obtained coefficients.	
def hypothesis(X, theta):
	h_X = np.dot(X, theta)
	
	return h_X



theta, prediction, cost = gradient_descent(X, y, LR = 0.062, iterations = 70000)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = np.asarray(train_test_split(X, y, test_size=0.25,random_state = 20))


import statsmodels.api as sm
X_train_sm = sm.add_constant(X)

from sklearn.metrics import r2_score
model_ohc = sm.OLS(y, X_train_sm).fit()

pred_ohc = model_ohc.predict(X_train_sm)
r2_ohc = r2_score(y, pred_ohc)
r2_ohc = r2_score(y, prediction)


