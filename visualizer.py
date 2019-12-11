from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas import DataFrame
from federatedPCA import merge,privateSAPCA,SMSULQ,SAPCA

data = read_csv('mnist.csv')
newdf = DataFrame(scale(data), index=data.index, columns=data.columns)
XMat = newdf.rename_axis('ID').values

XMat=XMat.T

# UOg, s, VOg = np.linalg.svd(XMat)
# n=min(XMat.shape)
# SOg = np.zeros((n, n))
# SOg[:n, :n] = np.diag(s)


#XMat1=XMat[:,:500]
#XMat2=XMat[:,500:1000]
# XMat3=XMat[:,1000:1500]

# U1, s1, V = np.linalg.svd(XMat1)
# n=min(XMat1.shape)
# S1 = np.zeros((n,n))
# S1[:n, :n] = np.diag(s1)

# U2, s2, V = np.linalg.svd(XMat2)
# n=min(XMat2.shape)
# S2 = np.zeros((n,n))
# S2[:n, :n] = np.diag(s2)

# U3, s3, V = np.linalg.svd(XMat3)
# n=min(XMat3.shape)
# S3 = np.zeros((n,n))
# S3[:n, :n] = np.diag(s3)

# UM1,SM1=merge(7,U1,S1,U2,S2)

#UMerged,SMerged=SMSULQ(7,XMat,730,.01,4)
rMerged,UMergedSa,SMergedSa=SAPCA(7,XMat[:,:5000],200,1.e-6,1)
rMer,UMerged,SMerged=privateSAPCA(7,XMat[:,:5000],200,1.e-6,1,.01,4)
print(rMer)
nPar=UMerged.T.dot(XMat)
#ogComp=UOg.T.dot(XMat)

# UDist=np.load('currentU.npy')
nComp=UMergedSa.T.dot(XMat)

# x=ogComp[0,:]
# y=ogComp[1,:]
# plt.scatter(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('PCA completo')
# plt.show()

x=-nComp[0,:]
y=nComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Distribuido')
plt.show()

x=nPar[0,:]
y=nPar[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Private SAPCA')
plt.show()
