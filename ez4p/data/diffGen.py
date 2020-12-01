import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pickle
import os

def SIAR(Z,t,theta):
    lamb = theta[0]
    alpha = theta[1]
    mu = theta[2]
    gamma = theta[3]

    S = Z[0]
    I = Z[1]
    A = Z[2]
    R = Z[3]
    StoI = Z[4]
    StoA = Z[5]
    ItoR = Z[6]


    N = S+I+A+R

    dS = - lamb*S*(A+alpha*I)/N 
    dI = - gamma*I + mu*lamb*S*(A+alpha*I)/N  
    dA = - gamma*A + (1-mu)*lamb*S*(A+alpha*I)/N
    dR = gamma*A + gamma*I

    dStoI = mu*lamb*S*(A+alpha*I)/N
    dStoA = (1-mu)*lamb*S*(A+alpha*I)/N
    dItoR = gamma*I

    return np.array([dS,dI,dA,dR,dStoI,dStoA,dItoR])

def genData(Z0,theta,T):
    
    t = np.linspace(0, T, T+1)
    data = spi.odeint(SIAR, Z0+[0,0,0] , t, args=(theta,))

    return data

def plotter(tm,tt,y,p1,p2,p3):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(tm, y[:, 0], label='Susceptible',color='teal')
    ax.plot(tm, y[:, 1], label='Infectious',color='blue')
    ax.plot(tm, y[:, 2], label='Asymptomatic',color='orange')
    ax.plot(tm, y[:, 3], label='Recovered',color='red')
    
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.title('Time Evolution of Epidemic')
    plt.xlabel('Time (in days)')
    plt.ylabel('Population')
    plt.grid()
    plt.savefig(p1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(tm, y[:, 4], label='Net Sympt. (S-I)',color='orange')
    ax.plot(tm, y[:, 5], label='Net Asympt. (S-A)',color='red')
    ax.plot(tm, y[:, 6], label='Net Recovered (I-R)',color='blue')
    
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.title('Time Evolution of Epidemic')
    plt.xlabel('Time (in days)')
    plt.ylabel('Population')
    plt.grid()
    plt.savefig(p2)


    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(tt, np.diff(y[:, 4]), label='New Infected (S-I)',color='orange')
    ax.plot(tt, np.diff(y[:, 5]), label='New Asymp. (S-A)',color='red')
    ax.plot(tt, np.diff(y[:, 6]), label='New Recovered (I-R)',color='blue')
    
    leg = ax.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.title('Time Evolution of Epidemic')
    plt.xlabel('Time (in days)')
    plt.ylabel('Population')
    plt.grid()
    plt.savefig(p3)

    return


def main(config):

    Z0 = [None]*4
    theta = [None]*4

    Z0[0] = config['S0']
    Z0[1] = config['I0']
    Z0[2] = config['A0']
    Z0[3] = config['R0']
    theta[0] = config['lamb']
    theta[1] = config['alpha']
    theta[2] = config['mu']
    theta[3] = config['gamma']
    T = config['horizon']
    
    Z0 = [str(z) for z in Z0]
    theta = [str(th) for th in theta]

    pkl_path = config['pkl_path']
    plot_path = config['plot_path']
    
    name = '-'.join(Z0) + '_' + '-'.join(theta)+'-'+str(T)    
    
    pkl_path = pkl_path + name + '.pkl'
    p1 = plot_path + name + '_main.png'
    p2 = plot_path + name + '_NetTrans.png'
    p3 = plot_path + name + '_Trans.png'

    

    Z0f = [float(z) for z in Z0]
    thetaf = [float(th) for th in theta]
    N = sum(Z0f)
    T = int(T)

    garbage = genData(Z0f,thetaf,T)
    Data = garbage[:,:4]
    NetTrans = garbage[:,4:]
    Trans = np.diff(NetTrans,axis=0)

    time_main = np.linspace(0,T,T+1)
    time_trans = np.linspace(0,T-1,T)
    plotter(time_main,time_trans,garbage/N,p1,p2,p3)
    
    Data = np.transpose(Data)
    Trans = np.transpose(Trans)
    
    params = {'lamb':thetaf[0],'alpha':thetaf[1],
                'mu':thetaf[2],'gamma':thetaf[3]}
    pickle.dump((Data, Trans, params), open(pkl_path, 'wb'))
    
    return 

if __name__ == "__main__":
    config = yaml.load(open('genConfig.yaml', 'r'), Loader=yaml.FullLoader)
    main(config)
    