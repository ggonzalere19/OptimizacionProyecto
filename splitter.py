from pandas import read_csv

X = read_csv('mnist.csv')

for i in range(2000,len(X.index),2000):
    dfAux=X[i-2000:i]
    dfAux.to_csv(index=False,path_or_buf='./mnist'+str(i)+".csv")

