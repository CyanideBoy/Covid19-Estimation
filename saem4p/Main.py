import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os
import torch
import matplotlib.pyplot as plt
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
BSIZE = int(config['batch'])


sData = Data[0,:].copy()
iData = Data[1,:].copy()    ## only iData accessible
aData = Data[2,:].copy()
rData = Data[3,:].copy()

iMax = np.max(iData)
dayMax = np.argmax(iData)

initDay = np.argmin((iData[:dayMax]-iMax/2)**2)
finalDay = dayMax + np.argmin((iData[dayMax:]-iMax/2)**2)


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
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############# PARAM INIT
lamb = torch.tensor([0.1]).double().to(device)  #0.2
lamb.requires_grad = True

alpha = torch.tensor([0.5]).double().to(device) #0.4
alpha.requires_grad = True

mu = torch.tensor([0.6]).double().to(device)    #0.7
mu.requires_grad = True

gamma = torch.tensor([0.1]).double().to(device)    #0.07
gamma.requires_grad = True

theta = [lamb,alpha,mu,gamma]


print('Horizon Length : ',H)
print('Initial Compartment Populations - S: {:.2f}, I: {:.2f}, A: {:.2f}, R: {:.2f}'.format(initCond[0],initCond[1],initCond[2],initCond[3]))
print('Parameters Initial Condition - Lambda: {:.2f}, Alpha: {:.2f}, Mu: {:.2f}, Gamma: {:.2f}'.format(theta[0].item(),theta[1].item(),theta[2].item(),theta[3].item()))

it = 0

#### Sim Loop
Q = 0
y = [0,0,0,0]
yp = [0,0,0,0]

opt1 = optim.Adam([{'params': theta[0], 'lr': eta[0]},
            {'params': theta[2], 'lr': eta[2]},
            {'params': theta[3], 'lr': eta[3]},
            ], lr=1e-2)
opt2 = optim.Adam([{'params': theta[0], 'lr': eta[0]}], lr=1e-2)
opt3 = optim.Adam([{'params': theta[1], 'lr': eta[1]}], lr=1e-2)

lambda1 = lambda epoch: 1/(1+epoch/decay_gd) if epoch < 80 else 0
lambda_alpha = lambda epoch: 1/(1+(epoch-80)/decay_gd) if epoch>=80 else 0
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda1) #lr_lambda= [lambda1,lambda_alpha,lambda1,lambda1])
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda= [lambda1,lambda_alpha,lambda1,lambda1])

s1 = lr_scheduler.LambdaLR(opt1, lr_lambda= lambda1)
s2 = lr_scheduler.LambdaLR(opt2, lr_lambda= lambda1)
s3 = lr_scheduler.LambdaLR(opt3, lr_lambda= lambda_alpha)

Ek = np.zeros((16,H,2))
A = np.zeros((16,H+1))
S = np.zeros((16,H+1))

while it < iters:
    
    opt1.zero_grad()
    opt2.zero_grad()
    opt3.zero_grad()
    
    h_len = H #np.random.randint(1,H)
    Ek = np.roll(Ek,1,axis=0)
    A = np.roll(A,1,axis=0)
    S = np.roll(S,1,axis=0)

    ek,a,s = randomSample(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],initCond,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)
    if it == 0:
        Ek = np.tile(ek,(16,1,1))
        S = np.tile(s,(16,1))
        A = np.tile(a,(16,1))
        
    else:
        Ek[0] = ek[0]
        S[0] = s[0]
        A[0] = a[0]
        

    LL = -H*getLogProb(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],torch.tensor(Ek).to(device),A,S,theta,initCond,T,D,it,decay_em,device)/(BSIZE*h_len)
    LL.backward()
    
    writer.add_scalars('Parameters', {'Lambda':theta[0].detach().cpu().numpy(),
                                        'Alpha':theta[1].detach().cpu().numpy(),
                                        'Mu':theta[2].detach().cpu().numpy(),
                                        'Gamma':theta[3].detach().cpu().numpy()}, it)
    writer.add_scalars('Log Likelihood', {'LL ': LL.detach().cpu().numpy()}, it)    
    writer.flush()
    
    opt2.zero_grad()
    opt3.zero_grad()
    
    if it%10 == 0:
        print('Iteration {} ongoing....'.format(it+1))
        print('Theta : ',[t.cpu().item() for t in theta])
        print('LL : ',LL.item())
        print('Gradients : ',[x.grad.item() for x in theta])
    opt1.step()

    PP = -H*getP1(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],torch.tensor(Ek).to(device),A,S,theta,initCond,T,D,it,decay_em,device)/(BSIZE*h_len)
    PP.backward()
    if it%10 == 0:
        print('Gradients : ',theta[0].grad.item(),theta[1].grad.item())
    opt2.step()
    opt3.step()

    with torch.no_grad():
        for x in theta:
            x += x.clamp_(0.01,0.99)-x

    if it%100 == 0:
        torch.save(theta,'weights/{}-iter-{}.pt'.format(SimName,it))
    it += 1
    s1.step()
    s2.step()
    s3.step()

writer.close()