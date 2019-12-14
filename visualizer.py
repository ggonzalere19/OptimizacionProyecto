from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas import DataFrame,Series
import scipy

# Como algunas componentes pueden tener signos volteados esta funcion sirve para que 
# las aproximaciones tengan el mismo signo que la original.
def checkSigns(UOriginal, UAprox):
    for i in range(UAprox.shape[1]):
        UAprox[:,i]=np.sign(UOg[0,i]*UAprox[0,i])*UAprox[:,i]
    return UAprox

# Se selecciona que ejemplo visualizar
dataSetName='normalData'
data = read_csv('./datasets/'+dataSetName+'.csv')
newdf = DataFrame(scale(data), index=data.index, columns=data.columns)
XMat = newdf.rename_axis('ID').values
XMat=XMat.T

# Se calcula la svd original para obtener PCA exacto
UOg, s, VOg = np.linalg.svd(XMat)

# Se leen los valores previamente calculados de las aproximaciones usando SAPCA y private-SAPCA
UPrivate=np.load('currentUPrivate'+dataSetName+'.npy')
USapca=np.load('currentU'+dataSetName+'.npy')
rPrivate=UPrivate.shape[1]
rSAPCA=USapca.shape[1]

# Se verifica que los signos en todas sean iguales
USapca=checkSigns(UOg,USapca)
UPrivate=checkSigns(UOg,UPrivate)

# Se calculan las distancias de Frobenius con respecto de PCA exacto.
distSapca=np.linalg.norm(UOg[:,:rSAPCA]-USapca)
distPrivate=np.linalg.norm(UOg[:,:rPrivate]-UPrivate)

# Se evaluan las componentes principales usando las matrices descritas anteriormente
privateComp=UPrivate.T.dot(XMat)
ogComp=UOg.T.dot(XMat)
SapcaComp=USapca.T.dot(XMat)

# Se grafican las primeras dos componentes para PCA exacto, SAPCA y private-SAPCA
x=ogComp[0,:]
y=ogComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA completo ' + dataSetName)
plt.show()

x=SapcaComp[0,:]
y=SapcaComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SAPCA '+ dataSetName+' rango: '+ str(rSAPCA)+ ' dist a U: '+ str(distSapca))
plt.show()

x=privateComp[0,:]
y=privateComp[1,:]
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Private SAPCA '+ dataSetName+' rango: '+ str(rPrivate)+ ' dist a U: '+ str(distPrivate))
plt.show()
