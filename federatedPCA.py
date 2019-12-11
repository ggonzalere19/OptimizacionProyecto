import numpy as np

def privateSAPCA(r,Y,b,alfa,beta,epsilon,delta):
    U=np.array([])
    S=np.array([])
    while(Y.size>0):
        if Y.shape[1]>b:
            Ys=Y[:,:b]
            Uk,Sk=SMSULQ(r,Ys,b,epsilon,delta)
            print Uk, Sk
            Y=np.delete(Y,np.s_[:b],axis=1)
        else:
            Uk,Sk = SMSULQ(r,Y,b,epsilon,delta)
            Y=np.delete(Y, np.s_[:Y.shape[1]], axis=1)
        if U.size!=0 and S.size!=0:
            U,S=merge(r,U,S,Uk,Sk)
        else:
            U,S=Uk,Sk
        #print('merged')
        r,U,S = RankAdjust(r,U,S,alfa,beta)
    return r,U,S

def SMSULQ(r,B,b,epsilon,delta):
    U=np.array([])
    S=np.array([])
    omega=CalculateOmega(B.shape[0],B.shape[1],epsilon,delta)
    BOg=B
    B=B.T
    while(B.size>0):
        #print 'smsulq'
        #print B.shape
        if B.shape[1]>b:
            #N = np.random.normal(0, omega*omega, (BOg.shape[0], b))
            N = np.zeros((BOg.shape[0],b))
            Bsi=B[:,:b]
            Bs=(1/b)*BOg.dot(Bsi)+N
            U,S=RSPCA(b,Bs,U,S)
            B=np.delete(B,np.s_[:b],axis=1)
        else:
            #N = np.random.normal(0, omega*omega, (BOg.shape[0], B.shape[1]))
            N = np.zeros((BOg.shape[0],B.shape[1]))
            Bs=(1/b)*BOg.dot(B)+N
            U,S = RSPCA(b,Bs,U,S)
            B=np.delete(B, np.s_[:B.shape[0]], axis=1)
    return U,S

def CalculateOmega(d,n,epsilon,delta):
    return ((d+1)/(n*epsilon))*np.sqrt(2*np.log((d*d+d)/(2*delta*np.sqrt(2*np.pi))))+1/(n*np.sqrt(epsilon))

def SAPCA(r,Y,b,alfa,beta):
    U=np.array([])
    S=np.array([])
    while(Y.size>0):
        if Y.shape[1]>b:
            Ys=Y[:,:b]
            U,S=RSPCA(r,Ys,U,S)
            Y=np.delete(Y,np.s_[:b],axis=1)
        else:
            U,S = RSPCA(r,Y,U,S)
            Y=np.delete(Y, np.s_[:Y.shape[1]], axis=1)
        r,U,S = RankAdjust(r,U,S,alfa,beta)
    return r,U,S

def RankAdjust(r,U,S,alfa,beta):
    c=CalculateC(S,r)
    if c>beta and U.shape[1]<r:
        oneVector=np.zeros((U.shape[0],1))
        oneVector[r]=1
        newS=np.zeros((S.shape[0]+1, S.shape[1]+1))
        newS[r,r]=1.e-3
        newS[:r,:r]=S
        return r+1,np.append(U,oneVector,axis=1),newS
    if c<alfa and r>1:
        return r-1,U[:,:r-1],S[:r-1,:r-1]
    return r, U,S

def CalculateC(S,r):
    sigmaR=S[0,0]
    sumSigma=0
    for i in range(r):
        sumSigma+=S[i,i]
    if sumSigma==0:
        return -1
    #print(sigmaR)
    #print(sumSigma)
    return sigmaR/sumSigma

def RSPCA(r,D,U,S):
    if U.size==0 and S.size==0:
        U,S,V = svdTrunc(r,D)
    else:
        U,S = merge(r,U,S,D,np.identity(D.shape[1]))
    return U,S

def merge(r,U1,S1,U2,S2):
    Z=U1.T.dot(U2)
    Q,R=np.linalg.qr(U2-U1.dot(Z))
    matAuxArriba = np.append(S1,Z.dot(S2),axis=1)
    RS2=R.dot(S2)
    matAux = np.append(matAuxArriba,np.append(np.zeros((RS2.shape[0],S1.shape[1])),RS2,axis=1),axis=0)
    U, S, V = svdTrunc(r,matAux)
    U = np.append(U1,Q,axis=1).dot(U)
    return U,S[:r,:r]

def svdTrunc(r,A):
    U, s, V = np.linalg.svd(A)
    S = np.zeros((r, r))
    S[:s.size, :s.size] = np.diag(s[:r])
    return U[:,:r],S[:r,:r],V[:r,:]
