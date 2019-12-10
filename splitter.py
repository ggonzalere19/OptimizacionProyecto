from pandas import read_csv

data = read_csv('wine.csv')
X = data.drop("quality",1)

for i in range(250,len(X.index),250):
    dfAux=X[i-250:i]
    dfAux.to_csv(index=False,path_or_buf='./wine'+str(i)+".csv")

