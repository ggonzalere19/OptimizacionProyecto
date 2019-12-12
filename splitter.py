from pandas import read_csv

X = read_csv('./datasets/normalData.csv')
blockSize=2000
scanned=0

for i in range(blockSize,len(X.index),blockSize):
    dfAux=X[i-blockSize:i]
    dfAux.to_csv(index=False,path_or_buf='./datasets/normalData'+str(i)+".csv")
    scanned+=blockSize

dfAux=X[scanned:]
dfAux.to_csv(index=False,path_or_buf='./datasets/normalData'+str(X.shape[0])+".csv")
