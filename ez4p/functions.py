from scipy.stats import binom, rv_discrete, truncnorm
import numpy as np
from math import ceil, floor
import torch.nn.functional as F
import torch

def randomSample(iObs, siTran, irTran, initCond, theta, T, D, B):
    
    N = np.sum(initCond)
    H = len(irTran)
    
    PA = D*theta[3]/T
    mu = theta[2]
    alpha = theta[1]
    lamb = theta[0]

    ek = -1*np.ones((B,H,2)) # SA AR
    ek[:,:,1] = np.random.binomial(T,PA,size=(B,H)) 
    
    S = np.tile(iObs.copy(),(B,1))
    A = np.tile(iObs.copy(),(B,1))
    S[:,0] = initCond[0]
    A[:,0] = initCond[2]

    for j in range(B):
        i = 0
        while i < H:
            PS = D*lamb*(alpha*iObs[i]+A[j,i])/(N*T)
            P = PS*(1-mu)/(1-mu*PS)

            Delta_S = S[j,i]/D
            Delta_A = A[j,i]/D
                
            dsi =  siTran[i]/Delta_S
            ek[j,i,0] = np.random.binomial(T-int(dsi),P) #(T-int(dsi))*P #

            A[j,i+1] = A[j,i] + Delta_S*ek[j,i,0] - Delta_A*ek[j,i,1]
            S[j,i+1] = S[j,i] - Delta_S*(ek[j,i,0] + int(dsi))
            i += 1 
    
    return ek,A,S

def getLogProb(iObs, siTran, irTran, ek, A, S, theta, initCond, T, D, device):
    N = np.sum(initCond)
    H = ek.shape[1]
    B = ek.shape[0]
    
    PI = (D*theta[3]/T).double()
    PA = (D*theta[3]/T).double()
    ll = 0
    
    S = torch.tensor(S).to(device)
    A = torch.tensor(A).to(device)
    
    esiTran = torch.zeros((B,len(siTran))).to(device).double() #np.tile(siTran.copy(),(B,1))    
    for i in range(len(siTran)):
        esiTran[:,i] = D*siTran[i]/S[:,i]

    C = 0.1
    #conf = np.exp(-C*np.arange(H))
    conf = 1/(1+np.arange(H))
    conf = H*conf/np.sum(conf)

    for i in range(H):
        PS = (D*theta[0]*(theta[1]*iObs[i]+A[:,i])/(N*T)).double()

        #### P1
        P1 = -T*F.kl_div(torch.log(torch.cat((PS,1-PS))), torch.cat(((ek[:,i,0]+esiTran[:,i])/T,1-(ek[:,i,0]+esiTran[:,i])/T)), reduction='sum')
        
        #### P2
        P2 = 0

        for j in range(B):
            if ek[j,i,0] == 0:
                P2 += (esiTran[j,i]*torch.log(theta[2]))[0]  #F.kl_div(torch.tensor([1,0]).log(), torch.cat((theta[2],1-theta[2])), reduction='sum')
            elif esiTran[j,i] == 0:
                P2 += (ek[j,i,0]*torch.log(1-theta[2]))[0]
            else:
                P2 += -(ek[j,i,0]+esiTran[j,i])*F.kl_div(torch.log(torch.cat((theta[2],1-theta[2]))), torch.tensor([esiTran[j,i]/(ek[j,i,0]+esiTran[j,i]),ek[j,i,0]/(ek[j,i,0]+esiTran[j,i])]).to(device), reduction='sum')
        
        #### P3
        X = torch.tensor([irTran[i]/iObs[i],1-(irTran[i]/iObs[i])]).to(device).double()
        Y = torch.cat((PI,1-PI))
        P3 = -B*T*(X * (X/Y).log()).sum()

        #### P4    
        X = torch.cat((ek[:,i,1]/T,1-(ek[:,i,1]/T)))
        Y = torch.zeros(2*B).to(device).double()
        Y[:B] = PA
        Y[B:] = 1-PA
        P4 = -T*(X * (X/Y).log()).sum()
        
        ll += conf[i]*(P1+P2+P3+P4)
    
    return ll