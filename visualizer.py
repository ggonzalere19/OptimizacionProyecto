from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas import DataFrame,Series
from federatedPCA import merge,privateSAPCA,SMSULQ,SAPCA
import scipy

dataSetName='wine'
data = read_csv('./datasets/'+dataSetName+'.csv')
data=data.drop('quality',axis=1)
newdf = DataFrame(scale(data), index=data.index, columns=data.columns)
nunique = newdf.apply(Series.nunique)
cols_to_drop = nunique[nunique == 1].index
newdf.drop(cols_to_drop, axis=1)

XMat = newdf.rename_axis('ID').values
XMat=XMat.T

UOg, s, VOg = np.linalg.svd(XMat)

UPrivate=np.load('currentUPrivate'+dataSetName+'.npy')
USapca=np.load('currentU'+dataSetName+'.npy')
rPrivate=UPrivate.shape[1]
rSAPCA=USapca.shape[1]

privateComp=UPrivate.T.dot(XMat)
ogComp=UOg.T.dot(XMat)
SapcaComp=USapca.T.dot(XMat)

x=-ogComp[0,:]
y=ogComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA completo ' + dataSetName)
plt.show()

x=-SapcaComp[0,:]
y=-SapcaComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SAPCA '+ dataSetName+' rango: '+ str(rSAPCA))
plt.show()

x=-privateComp[0,:]
y=privateComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Private SAPCA '+ dataSetName+' rango: '+ str(rPrivate))
plt.show()
