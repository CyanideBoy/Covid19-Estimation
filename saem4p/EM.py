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

    ek = -1*np.ones((B,H,2)) # dSA dAR
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
            S[j,i+1] = S[j,i] - Delta_S*(ek[j,i,0] + dsi)
            i += 1 
    
    return ek,A,S

def getLogProb(iObs, siTran, irTran, ek, A, S, theta, initCond, T, D, it, decay, device):
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
    
    #weights = np.exp(-2*np.arange(H))
    #weights = H*weights/np.sum(weights)
    
    if it>B:
        rates = 1/(1+(it+np.arange(B))/decay)
        rates = rates/np.sum(rates)
    else:
        rates = np.ones(B)/B

    for i in range(H):
        for j in range(B):
            PS = (D*theta[0]*(theta[1]*iObs[i]+A[j,i])/(N*T)).double()

            #### P1
            P1 = -T*F.kl_div(torch.log(torch.cat((PS,1-PS))), torch.tensor([(ek[j,i,0]+esiTran[j,i])/T,1-(ek[j,i,0]+esiTran[j,i])/T]).to(device), reduction='sum')
            
            #### P2
            if ek[j,i,0] == 0:
                P2 = (esiTran[j,i]*torch.log(theta[2]))[0]  #F.kl_div(torch.tensor([1,0]).log(), torch.cat((theta[2],1-theta[2])), reduction='sum')
            elif esiTran[j,i] == 0:
                P2 = (ek[j,i,0]*torch.log(1-theta[2]))[0]
            else:
                P2 = -(ek[j,i,0]+esiTran[j,i])*F.kl_div(torch.log(torch.cat((theta[2],1-theta[2]))), torch.tensor([esiTran[j,i]/(ek[j,i,0]+esiTran[j,i]),ek[j,i,0]/(ek[j,i,0]+esiTran[j,i])]).to(device), reduction='sum')
            
            #### P3
            X = torch.tensor([irTran[i]/iObs[i],1-(irTran[i]/iObs[i])]).to(device).double()
            Y = torch.cat((PI,1-PI))
            P3 = -T*(X * (X/Y).log()).sum()

            #### P4    
            X = torch.tensor([ek[j,i,1]/T,1-(ek[j,i,1]/T)]).to(device).double()
            Y = torch.cat((PA,1-PA))
            P4 = -T*(X * (X/Y).log()).sum()
        
            ll += (P1+P2+P3+P4)*rates[j]
    return ll

def getP1(iObs, siTran, irTran, ek, A, S, theta, initCond, T, D, it, decay, device):
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
    
    weights = np.exp(-5*np.arange(H))
    weights = H*weights/np.sum(weights)
    
    if it>B:
        rates = 1/(1+(it+np.arange(B))/decay)
        rates = rates/np.sum(rates)
    else:
        rates = np.ones(B)/B

    for i in range(H):
        for j in range(B):
            PS = (D*theta[0]*(theta[1]*iObs[i]+A[j,i])/(N*T)).double()

            #### P1
            P1 = -T*F.kl_div(torch.log(torch.cat((PS,1-PS))), torch.tensor([(ek[j,i,0]+esiTran[j,i])/T,1-(ek[j,i,0]+esiTran[j,i])/T]).to(device), reduction='sum')
            ll += P1*rates[j]*weights[i]
    return ll