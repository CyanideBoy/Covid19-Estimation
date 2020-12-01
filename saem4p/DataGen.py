import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def plotter(tm,tt,y,path):

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    gs.update(hspace=0.4)


    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
    ax1.plot(tt, np.diff(y[:, 4]), label='New Infected (S-I)',color='orange')
    ax1.plot(tt, np.diff(y[:, 5]), label='New Asymp. (S-A)',color='red')
    ax1.plot(tt, np.diff(y[:, 6]), label='New Recovered (I-R)',color='blue')
    leg = ax1.legend(loc='upper left',fontsize='xx-small')
    plt.title('Daily Transitions')
    plt.grid()

    ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
    ax2.plot(tm, y[:, 4], label='Net Sympt. (S-I)',color='orange')
    ax2.plot(tm, y[:, 5], label='Net Asympt. (S-A)',color='red')
    ax2.plot(tm, y[:, 6], label='Net Recovered (I-R)',color='blue')
    leg = ax2.legend(fontsize='xx-small')
    plt.grid()
    plt.title('Total Transitions Since Start')

    ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
    ax3.plot(tm, y[:, 0], label='Susceptible',color='teal')
    ax3.plot(tm, y[:, 1], label='Infectious',color='blue')
    ax3.plot(tm, y[:, 2], label='Asymptomatic',color='orange')
    ax3.plot(tm, y[:, 3], label='Recovered',color='red') 
    leg = ax3.legend(fontsize='small')
    
    
    plt.title('Compartment Populations')
    plt.xlabel('Time (in days)')
    plt.ylabel('Population')
    plt.grid()

    plt.savefig(path)

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
    plt_path = config['plt_path']
    
    name = '-'.join(Z0) + '_' + '-'.join(theta)+'-'+str(T)    
    
    pkl_path = 'data/' + pkl_path + name + '.pkl'
    plt_path = 'data/' + plt_path + name + '.png'
    
    

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
    plotter(time_main,time_trans,garbage/N,plt_path)
    
    Data = np.transpose(Data)
    Trans = np.transpose(Trans)
    
    params = {'lamb':thetaf[0],'alpha':thetaf[1],
                'mu':thetaf[2],'gamma':thetaf[3]}
    pickle.dump((Data, Trans, params), open(pkl_path, 'wb'))
    
    return 

if __name__ == "__main__":
    config = yaml.load(open('Config-DataGen.yaml', 'r'), Loader=yaml.FullLoader)
    main(config)
    