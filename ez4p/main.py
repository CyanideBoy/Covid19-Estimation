import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os
import torch
import matplotlib.pyplot as plt
from functions import randomSample, getLogProb
import pandas as pd
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import datetime
#import torch_optimizer as optim
import torch.optim as optim
from torch.optim import lr_scheduler


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('Running on',device)


config = yaml.load(open('simConfig.yaml', 'r'), Loader=yaml.FullLoader)
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

decay = float(config['decay'])
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


# d1 = pd.DataFrame({'sData':sData[initDay:finalDay],'iData': iData[initDay:finalDay],'aData':aData[initDay:finalDay]})
# d3 = pd.DataFrame({'si':Trans[0,initDay:finalDay], 'sa':Trans[1,initDay:finalDay], 'ir':Trans[2,initDay:finalDay]})
# dataset = pd.concat([d1,d3], ignore_index=True, axis=1)
# dataset.to_csv('var.csv')

############## Seeding
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############# PARAM INIT
lamb = torch.tensor([0.05]).double().to(device)  #0.2
lamb.requires_grad = True

alpha = torch.tensor([0.5]).double().to(device) #0.4
alpha.requires_grad = True

mu = torch.tensor([0.5]).double().to(device)    #0.7
mu.requires_grad = True

gamma = torch.tensor([0.1]).double().to(device)    #0.07
gamma.requires_grad = True

theta = [lamb,alpha,mu,gamma]


####
print(H)
print(iObs.shape)
print(irTran.shape)
print(N)
print(initCond)
print([x.item() for x in theta])

it = 0
####  Resume Sim
if bool(config['resume']):
    f = []
    for (a,b,c) in os.walk('weights'):
        f.extend(c)
        break
    
    if len(f)==0:
        it = 0
    else:
        f = [x[:-3] for x in f]
        g = [int(x.split('-')[-1]) for x in f]
        it = max(g)
        w_name = '-'.join(f[0].split('-')[:-1]+[str(it)])+'.pt'    
        theta = torch.load('weights/'+w_name)

#### Sim Loop

optimizer = optim.Adam([{'params': theta[0], 'lr': eta[0]},
            {'params': theta[1], 'lr': eta[1]},
            {'params': theta[2], 'lr': eta[2]},
            {'params': theta[3], 'lr': eta[3]},
            ], lr=1e-2)

lambda1 = lambda epoch: 1/(1+epoch/decay)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda1)

while it < iters:
    
    optimizer.zero_grad()
    #h_len= 5
    h_len = np.random.randint(1,H)
    #h_len  = min(H,2 + int(it*((H-2)/150)))

    #h_len = H

    ek,A,S = randomSample(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],initCond,[x.detach().cpu().numpy() for x in theta],T,D,BSIZE)

    LL = -H*getLogProb(iObs[:h_len+1],siTran[:h_len],irTran[:h_len],torch.tensor(ek).to(device),A,S,theta,initCond,T,D,device)/(BSIZE*h_len)
    LL.backward()
    
    
    if it%10 == 0:
        print('Iteration {} ongoing....'.format(it+1))
        print('Theta : ',[t.cpu().item() for t in theta])
        print('LL : ',LL.item())
        
        
    writer.add_scalars('Parameters', {'Lambda':theta[0].detach().cpu().numpy(),
                                        'Alpha':theta[1].detach().cpu().numpy(),
                                        'Mu':theta[2].detach().cpu().numpy(),
                                        'Gamma':theta[3].detach().cpu().numpy()}, it)
    writer.add_scalars('Log Likelihood', {'LL ': LL.detach().cpu().numpy()}, it)    
    writer.flush()

    optimizer.step()
    with torch.no_grad():
        for x in theta:
            x += x.clamp_(0.02,0.98)-x
    
    if it%100 == 0:
        torch.save(theta,'weights/{}-iter-{}.pt'.format(SimName,it))
    it += 1
    scheduler.step()

writer.close()
'''
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
t = list(range(len(A[0,:])))

ax.plot(t, np.mean(S,axis=0), label='Pred Sus',color='teal')
ax.plot(t, iObs, label='Infectious',color='blue')
ax.plot(t, np.mean(A,axis=0), label='Pred Asymp',color='orange')
ax.plot(t, sData[initDay:finalDay+1], label='GT Sus',color='black')
ax.plot(t, aData[initDay:finalDay+1], label='GT Asymp',color='yellow')

leg = ax.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)
plt.title('Time Evolution of Epidemic')
plt.xlabel('Time (in days)')
plt.ylabel('Population')
plt.grid()
plt.savefig('tp1.png')
'''