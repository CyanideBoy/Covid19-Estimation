import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from EM import randomSample, getLogProb
import pandas as pd
from tensorboardX import SummaryWriter
import datetime
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)


config = yaml.load(open('Config-Sim.yaml', 'r'), Loader=yaml.FullLoader)

D1, T1, TP1 = pickle.load(open(config['dp1'], 'rb'))
D2, T2, TP2 = pickle.load(open(config['dp2'], 'rb'))
D3, T3, TP3 = pickle.load(open(config['dp3'], 'rb'))
D4, T4, TP4 = pickle.load(open(config['dp4'], 'rb'))

writer = SummaryWriter(log_dir=config['log_path']+'/'+datetime.datetime.now().strftime("%d-%H%M"))

T = int(config['T'])    # Discretization of day
D = int(config['D'])    # Discretization of people

eta1 = float(config['eta1'])
eta2 = float(config['eta2'])
eta3 = float(config['eta3'])
eta4 = float(config['eta4'])
eta = [eta1,eta2,eta3,eta4]

decay_gd = float(config['decay-gd'])
decay_em = float(config['decay-em'])
iters = int(config['iters'])
BSIZE = 1
SAEM = int(config['saem'])
F_NAME = config['name']
seed = int(config['seed'])

################################
iD1 = D1[1,:].copy()   
iMax = np.max(iD1)
dayMax = np.argmax(iD1)
initDay = np.argmin((iD1[:dayMax]-iMax/4)**2)
finalDay = dayMax + np.argmin((iD1[dayMax:]-iMax/4)**2)
H1 = finalDay - initDay
iD1 = iD1[initDay:finalDay+1]
siT1 = T1[0,initDay:finalDay]
irT1 = T1[2,initDay:finalDay]
iC1 = D1[:,initDay].copy()
################
iD2 = D2[1,:].copy()   
iMax = np.max(iD2)
dayMax = np.argmax(iD2)
initDay = np.argmin((iD2[:dayMax]-iMax/4)**2)
finalDay = dayMax + np.argmin((iD2[dayMax:]-iMax/4)**2)
H2 = finalDay - initDay
iD2 = iD2[initDay:finalDay+1]
siT2 = T2[0,initDay:finalDay]
irT2 = T2[2,initDay:finalDay]
iC2 = D2[:,initDay].copy()
###############
iD3 = D3[1,:].copy()   
iMax = np.max(iD3)
dayMax = np.argmax(iD3)
initDay = np.argmin((iD3[:dayMax]-iMax/4)**2)
finalDay = dayMax + np.argmin((iD3[dayMax:]-iMax/4)**2)
H3 = finalDay - initDay
iD3 = iD3[initDay:finalDay+1]
siT3 = T3[0,initDay:finalDay]
irT3 = T3[2,initDay:finalDay]
iC3 = D3[:,initDay].copy()
#################################################
iD4 = D4[1,:].copy()   
iMax = np.max(iD4)
dayMax = np.argmax(iD4)
initDay = 0 #np.argmin((iD4[:dayMax]-iMax/4)**2)
finalDay = dayMax + np.argmin((iD4[dayMax:]-iMax/4)**2)
H4 = finalDay - initDay
iD4 = iD4[initDay:finalDay+1]
siT4 = T4[0,initDay:finalDay]
irT4 = T4[2,initDay:finalDay]
iC4 = D4[:,initDay].copy()
###################################
N = np.sum(iC1)

############## Seeding
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############# PARAM INIT
lamb = torch.tensor([0.01]).double().to(device)  #0.2
lamb.requires_grad = True

alpha = torch.tensor([np.random.random_sample()]).double().to(device) #0.4
alpha.requires_grad = True

mu = torch.tensor([0.5]).double().to(device)    #0.7
mu.requires_grad = True

gamma = torch.tensor([np.random.random_sample()]).double().to(device)    #0.07
gamma.requires_grad = True

theta = [lamb,alpha,mu,gamma]


print('Horizon Length : ',H1,H2,H3,H4)
print('Parameters Initial Condition - Lambda: {:.2f}, Alpha: {:.2f}, Mu: {:.2f}, Gamma: {:.2f}'.format(theta[0].item(),theta[1].item(),theta[2].item(),theta[3].item()))

it = 0
lossD = np.zeros(iters)
thetaD = np.zeros((4,iters))

#### Sim Loop
optR = optim.Adam([{'params': theta[0], 'lr': eta[0]},
            {'params': theta[2], 'lr': eta[2]},
            {'params': theta[3], 'lr': eta[3]},
            ], lr=1e-2)
optA = optim.Adam([{'params': theta[1], 'lr': eta[1]}], lr=1e-2)

lambdaR = lambda epoch: 1/(1+epoch/decay_gd) #if epoch<=100 else 0
lambdaA = lambda epoch: 1/(1+epoch/decay_gd) #if epoch >= 50 else 0

sR = lr_scheduler.LambdaLR(optR, lr_lambda= lambdaR)
sA = lr_scheduler.LambdaLR(optA, lr_lambda= lambdaA)

Ek1 = np.zeros((SAEM,H1,2))
A1 = np.zeros((SAEM,H1+1))
S1 = np.zeros((SAEM,H1+1))
Ek2 = np.zeros((SAEM,H2,2))
A2 = np.zeros((SAEM,H2+1))
S2 = np.zeros((SAEM,H2+1))
Ek3 = np.zeros((SAEM,H3,2))
A3 = np.zeros((SAEM,H3+1))
S3 = np.zeros((SAEM,H3+1))
Ek4 = np.zeros((SAEM,H4,2))
A4 = np.zeros((SAEM,H4+1))
S4 = np.zeros((SAEM,H4+1))

while it < iters:

    optR.zero_grad()
    optA.zero_grad()
    
    ek1,a1,s1 = randomSample(iD1,siT1,irT1,iC1,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)
    ek2,a2,s2 = randomSample(iD2,siT2,irT2,iC2,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)
    ek3,a3,s3 = randomSample(iD3,siT3,irT3,iC3,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)
    ek4,a4,s4 = randomSample(iD4,siT4,irT4,iC4,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)

    Ek1[0] = ek1[0]
    S1[0] = s1[0]
    A1[0] = a1[0]
    Ek2[0] = ek2[0]
    S2[0] = s2[0]
    A2[0] = a2[0]
    Ek3[0] = ek3[0]
    S3[0] = s3[0]
    A3[0] = a3[0]
    Ek4[0] = ek4[0]
    S4[0] = s4[0]
    A4[0] = a4[0]

    LL = 0
    LL += -getLogProb(iD1,siT1,irT1,torch.tensor(Ek1).to(device),A1,S1,theta,iC1,T,D,it,decay_em,device)/(4*BSIZE*SAEM)
    LL += -getLogProb(iD2,siT2,irT2,torch.tensor(Ek2).to(device),A2,S2,theta,iC2,T,D,it,decay_em,device)/(4*BSIZE*SAEM)
    LL += -getLogProb(iD3,siT3,irT3,torch.tensor(Ek3).to(device),A3,S3,theta,iC3,T,D,it,decay_em,device)/(4*BSIZE*SAEM)
    LL += -getLogProb(iD4,siT4,irT4,torch.tensor(Ek4).to(device),A4,S4,theta,iC4,T,D,it,decay_em,device)/(4*BSIZE*SAEM)
    LL.backward()

    lossD[it] = LL.cpu().item()
    thetaD[:,it] = np.array([t.cpu().item() for t in theta])

    writer.add_scalars('Parameters', {'Lambda':theta[0].detach().cpu().numpy(),
                                        'Alpha':theta[1].detach().cpu().numpy(),
                                        'Mu':theta[2].detach().cpu().numpy(),
                                        'Gamma':theta[3].detach().cpu().numpy()}, it)
    writer.add_scalars('Log Likelihood', {'LL ': LL.detach().cpu().numpy()}, it)    
    writer.flush()
    
    if it%500 == 0:
        print('---------- Iteration {} ongoing -------'.format(it+1))
        print('Theta : ',thetaD[:,it])
        print('LL : ',LL.item())
        print('Gradients : ',[x.grad.item() for x in theta])
    
    optR.step()
    optA.step()

    with torch.no_grad():
        for x in theta:
            x += x.clamp_(0.01,0.99)-x

    it += 1
    
    sR.step()
    sA.step()
    
writer.close()

### Print Final Values
print('Theta : ',[t.cpu().item() for t in theta])
with open('results/'+F_NAME+'.txt','w') as f:
    for i in range(len(theta)):
        f.write(str(theta[i].cpu().item())+'\n')
        print(theta[i].cpu().item())

t = np.arange(iters)
############# LL PLOT
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.semilogy(t,lossD)    
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood (on log scale)')
plt.savefig('results/'+F_NAME+'_LL.png')


############ PARAM PLOT
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(t, thetaD[0,:], label=r'$\lambda$',color='black')
plt.plot(t, 0.2*np.ones(iters), '--', label=None, color='black')

plt.plot(t, thetaD[1,:], label=r'$\alpha$',color='red')
plt.plot(t, 0.4*np.ones(iters), '--', label=None, color='red')

plt.plot(t, thetaD[2,:], label=r'$\mu$',color='blue')
plt.plot(t, 0.7*np.ones(iters), '--', label=None, color='blue')

plt.plot(t, thetaD[3,:], label=r'$\gamma$',color='orange')
plt.plot(t, 0.07*np.ones(iters), '--', label=None, color='orange')

plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Parameter Values')

plt.legend(prop={'size': 12})
plt.savefig('results/'+F_NAME+'_params.png')

############ CONTOUR PLOT
def cont(a,l,MU):
    return l*(a*MU + (1-MU))/MU
TRU = cont(0.4,0.2,0.7)

alph = np.linspace(0 , 1, 100)
lamb = np.linspace(0 , 1, 100)
A, L = np.meshgrid(alph, lamb)
z = np.array([cont(x,y,0.7) for (x,y) in zip(np.ravel(A),np.ravel(L))])
Z = z.reshape(A.shape)

a = thetaD[1,:]
l = thetaD[0,:]

fig = plt.figure(figsize=(20, 10))
c = plt.pcolor(A, L, Z )
plt.jet()
plt.colorbar(c,orientation='vertical')
plt.contour(A,L,Z,[TRU],linestyles='dashed',colors=['black'])
plt.plot(a,l,markerfacecolor='r', markeredgecolor='r', marker='x', color='black', linewidth=1, markersize=6)
plt.xlabel(r'A $\alpha$')
plt.ylabel(r'$\lambda$')
plt.title('Tracking the parameters on the contour')
plt.savefig('results/'+F_NAME+'_cont.png')
