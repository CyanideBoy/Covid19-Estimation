import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from EM import randomSample, getLogProb, getP1
import pandas as pd
from tensorboardX import SummaryWriter
import datetime
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)


config = yaml.load(open('Config-Sim.yaml', 'r'), Loader=yaml.FullLoader)
Data, Trans, TrueParams = pickle.load(open(config['data_path'], 'rb'))

SimName = config['data_path'].split('/')[-1][:-4]
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

sData = Data[0,:].copy()
iData = Data[1,:].copy()    ## only iData accessible
aData = Data[2,:].copy()
rData = Data[3,:].copy()

iMax = np.max(iData)
dayMax = np.argmax(iData)

initDay = np.argmin((iData[:dayMax]-iMax/6)**2)
finalDay = dayMax + np.argmin((iData[dayMax:]-iMax/6)**2)

############# Data to be considered
H = finalDay - initDay
iObs = iData[initDay:finalDay+1]
siTran = Trans[0,initDay:finalDay]
saTran = Trans[1,initDay:finalDay]  # Not part of dataset for training
irTran = Trans[2,initDay:finalDay]
initCond = Data[:,initDay].copy()
N = np.sum(initCond)

#d1 = pd.DataFrame({'sData':sData[initDay:finalDay],'iData': iData[initDay:finalDay],'aData':aData[initDay:finalDay]})
#d3 = pd.DataFrame({'si':Trans[0,initDay:finalDay], 'sa':Trans[1,initDay:finalDay], 'ir':Trans[2,initDay:finalDay]})
#dataset = pd.concat([d1,d3], ignore_index=True, axis=1)
#dataset.to_csv('var.csv')

############## Seeding
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############# PARAM INIT
lamb = torch.tensor([0.2]).double().to(device)  #0.2
lamb.requires_grad = True

#alpha = torch.tensor([np.random.random_sample()]).double().to(device) #0.4
alpha = torch.tensor([0.4]).double().to(device) #0.4
alpha.requires_grad = True

mu = torch.tensor([0.7]).double().to(device)    #0.7
mu.requires_grad = True

#gamma = torch.tensor([np.random.random_sample()]).double().to(device)    #0.07
gamma = torch.tensor([0.07]).double().to(device)    #0.07
gamma.requires_grad = True

theta = [lamb,alpha,mu,gamma]


print('Horizon Length : ',H)
print('Initial Compartment Populations - S: {:.2f}, I: {:.2f}, A: {:.2f}, R: {:.2f}'.format(initCond[0],initCond[1],initCond[2],initCond[3]))
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

Ek = np.zeros((SAEM,H,2))
A = np.zeros((SAEM,H+1))
S = np.zeros((SAEM,H+1))

while it < iters:

    optR.zero_grad()
    optA.zero_grad()
    
    h_len = H #np.random.randint(1,H)
    Ek = np.roll(Ek,1,axis=0)
    A = np.roll(A,1,axis=0)
    S = np.roll(S,1,axis=0)

    ek,a,s = randomSample(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],initCond,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)
    
    if it == 0:
        Ek = np.tile(ek,(SAEM,1,1))
        S = np.tile(s,(SAEM,1))
        A = np.tile(a,(SAEM,1)) 
    else:
        Ek[0] = ek[0]
        S[0] = s[0]
        A[0] = a[0]
        
    LL = -H*getLogProb(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],torch.tensor(Ek).to(device),A,S,theta,initCond,T,D,it,decay_em,device)/(BSIZE*SAEM*h_len)
    LL.backward()
    lossD[it] = LL.cpu().item()
    thetaD[:,it] = np.array([t.cpu().item() for t in theta])

    writer.add_scalars('Parameters', {'Lambda':theta[0].detach().cpu().numpy(),
                                        'Alpha':theta[1].detach().cpu().numpy(),
                                        'Mu':theta[2].detach().cpu().numpy(),
                                        'Gamma':theta[3].detach().cpu().numpy()}, it)
    writer.add_scalars('Log Likelihood', {'LL ': LL.detach().cpu().numpy()}, it)    
    writer.flush()
    
    if it%100 == 0:
        print('---------- Iteration {} ongoing -------'.format(it+1))
        print('Theta : ',thetaD[:,it])
        print('LL : ',LL.item())
        print('Gradients : ',[x.grad.item() for x in theta])
    
    optR.step()
    optA.step()

    '''
    PP = -H*getP1(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],torch.tensor(Ek).to(device),A,S,theta,initCond,T,D,it,decay_em,device)/(BSIZE*SAEM*h_len)
    PP.backward()
    if it%10 == 0:
        print('Gradients : ',theta[1].grad.item())
    optA.step()
    '''

    with torch.no_grad():
        for x in theta:
            x += x.clamp_(0.01,0.99)-x

    if it%100 == 0:
        torch.save(theta,'weights/{}-iter-{}.pt'.format(SimName,it))
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