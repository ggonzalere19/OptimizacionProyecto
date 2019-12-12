from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas import DataFrame,Series
from federatedPCA import merge,privateSAPCA,SMSULQ,SAPCA
import scipy

#XMat=np.load('normalData.npy')
dataSetName='normalData'
data = read_csv('./datasets/'+dataSetName+'.csv')
#data=data.drop('quality',axis=1)
newdf = DataFrame(scale(data), index=data.index, columns=data.columns)
nunique = newdf.apply(Series.nunique)
cols_to_drop = nunique[nunique == 1].index
newdf.drop(cols_to_drop, axis=1)

XMat = newdf.rename_axis('ID').values
XMat=XMat.T

UOg, s, VOg = np.linalg.svd(XMat)

#rSapca,USapca,SSapca=SAPCA(8,XMat,200,1.e-6,.1)
#rPrivate,UPrivate,SPrivate=privateSAPCA(8,XMat,XMat.shape[0],1.e-6,.5,4,.5)
UPrivate=np.load('currentUPrivate.npy')
USapca=np.load('currentU.npy')

privateComp=UPrivate.T.dot(XMat)
ogComp=UOg.T.dot(XMat)
SapcaComp=USapca.T.dot(XMat)

x=-ogComp[0,:]
y=ogComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA completo')
plt.show()

x=-SapcaComp[0,:]
y=SapcaComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SAPCA')
plt.show()

x=-privateComp[0,:]
y=privateComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Private SAPCA')
plt.show()
