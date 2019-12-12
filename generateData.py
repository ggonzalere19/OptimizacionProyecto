from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas import DataFrame,Series
from federatedPCA import merge,privateSAPCA,SMSULQ,SAPCA
import scipy

alfaMat=.5
d=50
n=10000

S=np.random.normal(0,1,(d,d))
S=scipy.linalg.orth(S)
lamb=np.zeros((d,d))
for i in range(d):
    lamb[i,i]=np.power(i+1,-alfaMat)
cov=S.T.dot(lamb).dot(S)

X=np.random.multivariate_normal(np.zeros((d)),cov).reshape(d,1)
for i in range(1,n):
    X=np.append(X,np.random.multivariate_normal(np.zeros((d)),cov).reshape(d,1),axis=1)

X=X.T

DataFrame(X).to_csv("normalData.csv")
