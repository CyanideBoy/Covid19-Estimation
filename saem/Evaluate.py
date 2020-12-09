import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from numpy import linalg as LA
import sklearn.metrics as metrics

def SIAR(Z,t,theta):
    lamb = theta[0]
    alpha = theta[1]
    mu = theta[2]
    gamma = theta[3]

    S = Z[0]
    I = Z[1]
    A = Z[2]
    R = Z[3]
    
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

def plotter(t,A,P,path):

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    gs.update(hspace=0.4)

    fig = plt.figure()

    ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
    ax1.plot(t, A[:,0], '--', label='GT',color='gray')
    ax1.plot(t, P[:,0], label='Pred',color='gray')
    plt.title('S Compartment')
    plt.grid()

    ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
    ax2.plot(t, A[:,1], '--', label='GT',color='blue')
    ax2.plot(t, P[:,1], label='Pred',color='blue')
    plt.title('I Compartment')
    plt.grid()

    ax3 = fig.add_subplot(gs[1, 0]) # row 1, col 0
    ax3.plot(t, A[:,2], '--', label='GT',color='red')
    ax3.plot(t, P[:,2], label='Pred',color='red')
    plt.title('A Compartment')
    plt.grid()

    ax4 = fig.add_subplot(gs[1, 1]) # row 0, col 1
    ax4.plot(t, A[:,3], '--', label='GT',color='orange')
    ax4.plot(t, P[:,3], label='Pred',color='orange')
    plt.title('R Compartment')
    plt.grid()

    #plt.title('Compartment Populations')
    #plt.xlabel('Time (in days)')
    #plt.ylabel('Population')
    #plt.grid()

    plt.savefig(path)
    return

def plot_corr(aH,tH):

    fig = plt.figure()
    plt.plot(tH, aH)
    plt.title('Auto-correlation')
    plt.grid(True)
    plt.xlabel('Time (in days)')
    plt.ylabel('Population')
    plt.savefig('results/autoC.png')
    return

def RMSE(x):
    return np.sqrt(np.mean(np.square(x)))

def main(config, thetaP):

    Z0 = [None]*4
    thetaA = [None]*4

    Z0[0] = config['S0']
    Z0[1] = config['I0']
    Z0[2] = config['A0']
    Z0[3] = config['R0']
    thetaA[0] = config['lamb']
    thetaA[1] = config['alpha']
    thetaA[2] = config['mu']
    thetaA[3] = config['gamma']
    T = config['horizon']
    
    Z0 = [float(z) for z in Z0]
    thetaA = [float(th) for th in thetaA]
    N = sum(Z0)
    T = int(T)

    gA = genData(Z0,thetaA,T)
    DataAF = gA[:,:4]
    #gP = genData(Z0,thetaP,T)
    #DataPF = gP[:,:4]
    
    ## Compute Window Used in the Experiment
    iMax = np.max(DataAF[:,1])
    dayMax = np.argmax(DataAF[:,1])
    initDay = np.argmin((DataAF[:dayMax,1]-iMax/2)**2)
    finalDay = dayMax + np.argmin((DataAF[dayMax:,1]-iMax/2)**2)
    H = finalDay-initDay
    print('Horizon Length :' ,H)

    gA = genData(list(DataAF[initDay,:]),thetaA,H)
    DataAH = gA[:,:4]
    gP = genData(list(DataAF[initDay,:]),thetaP,H)
    DataPH = gP[:,:4]

    '''
    ## Compute RMSE
    rmseF_S = f_RMSE(DataAF[:,0] - DataPF[:,0])
    rmseF_I = f_RMSE(DataAF[:,1] - DataPF[:,1])
    rmseF_A = f_RMSE(DataAF[:,2] - DataPF[:,2])
    rmseF_R = f_RMSE(DataAF[:,3] - DataPF[:,3])

    rmseH_S = f_RMSE(DataAH[:,0] - DataPH[:,0])
    rmseH_I = f_RMSE(DataAH[:,1] - DataPH[:,1])
    rmseH_A = f_RMSE(DataAH[:,2] - DataPH[:,2])
    rmseH_R = f_RMSE(DataAH[:,3] - DataPH[:,3])

    ## Compute %RMSE
    A = 10
    B = -10
    rmseF_pS = f_RMSE((DataAF[A:B,0] - DataPF[A:B,0])/DataAF[A:B,0])
    rmseF_pI = f_RMSE((DataAF[A:B,1] - DataPF[A:B,1])/DataAF[A:B,1])
    rmseF_pA = f_RMSE((DataAF[A:B,2] - DataPF[A:B,2])/DataAF[A:B,2])
    rmseF_pR = f_RMSE((DataAF[A:B,3] - DataPF[A:B,3])/DataAF[A:B,3])
    '''
    rmseH_pS = RMSE((DataAH[:,0] - DataPH[:,0])/DataAH[:,0])
    rmseH_pI = RMSE((DataAH[:,1] - DataPH[:,1])/DataAH[:,1])
    rmseH_pA = RMSE((DataAH[:,2] - DataPH[:,2])/DataAH[:,2])
    rmseH_pR = RMSE((DataAH[:,3] - DataPH[:,3])/DataAH[:,3])
    
    ## AutoCorr
    #eH = (DataAH[:,0] - DataPH[:,0])/(100+DataAH[:,0])
    #eH = eH[2:-2]/eH[2:-2] #np.convolve(eH,np.array([1,3,5,3,1])/13,'valid')
    #eH = eH - eH.mean()
    #aH = np.correlate(eH,eH,mode='full')
    #print(eH.shape)
    #aHmid = len(eH)-1
    #tH = np.arange(len(aH))-aHmid
    #plot_corr(aH,tH)

    #eF = DataAF[A:B,0] - DataPF[A:B,0]
    #eF = eF[2:-2]/DataAF[A+2:B-2,0] #np.convolve(eF,np.array([1,3,5,3,1])/13,'valid')
    #eF = eF - eF.mean()
    #aF = np.correlate(eF,eF,mode='full')
    #aFmid = len(eF)-1
    #tF = np.arange(len(aF))-aFmid
    #print(eF.mean())
    
    '''
    ## Convert to Single Array
    eqAF = (DataAF[:,0]/N) + N*(DataAF[:,1]/N) + N*N*(DataAF[:,2]/N) + N*N*N*(DataAF[:,3]/N)  
    eqPF = (DataPF[:,0]/N) + N*(DataPF[:,1]/N) + N*N*(DataPF[:,2]/N) + N*N*N*(DataPF[:,3]/N)
    eqAH = (DataAH[:,0]/N) + N*(DataAH[:,1]/N) + N*N*(DataAH[:,2]/N) + N*N*N*(DataAH[:,3]/N)  
    eqPH = (DataPH[:,0]/N) + N*(DataPH[:,1]/N) + N*N*(DataPH[:,2]/N) + N*N*N*(DataPH[:,3]/N)
    
    MIF = metrics.mutual_info_score(eqAF,eqPF)
    MIF_max = metrics.mutual_info_score(eqAF,eqAF)
    AMIF = metrics.adjusted_mutual_info_score(eqAF,eqPF)

    MIH = metrics.mutual_info_score(eqAH,eqPH)
    MIH_max = metrics.mutual_info_score(eqAH,eqAH)
    AMIH = metrics.adjusted_mutual_info_score(eqAH,eqPH)
    '''
    ## PRINT METRICS
    #print('RMSE Full ',[rmseF_S,rmseF_I,rmseF_A,rmseF_R])
    #print('RMSE Half ',[rmseH_S,rmseH_I,rmseH_A,rmseH_R])
    #print('RMSE Percent Full ',[rmseF_pS,rmseF_pI,rmseF_pA,rmseF_pR])
    print('RMSE Percent Half ',[rmseH_pS,rmseH_pI,rmseH_pA,rmseH_pR])
    
    #Cross Corr - 
    print('Normalized Cross Corr S',np.sum(DataAH[:,0]*DataPH[:,0])/(len(DataAH[:,0])*RMSE(DataAH[:,0])*RMSE(DataPH[:,0])))
    print('Normalized Cross Corr I',np.sum(DataAH[:,1]*DataPH[:,1])/(len(DataAH[:,1])*RMSE(DataAH[:,1])*RMSE(DataPH[:,1])))
    print('Normalized Cross Corr A',np.sum(DataAH[:,2]*DataPH[:,2])/(len(DataAH[:,2])*RMSE(DataAH[:,2])*RMSE(DataPH[:,2])))
    print('Normalized Cross Corr R',np.sum(DataAH[:,3]*DataPH[:,3])/(len(DataAH[:,3])*RMSE(DataAH[:,3])*RMSE(DataPH[:,3])))
    '''
    print("------")
    print('MI Full: ',MIF,' Max MI=',MIF_max)
    print('MI Half: ',MIH,' Max MI=',MIH_max)
    print('AMI Full: ',AMIF)
    print('AMI Half: ',AMIH)
    '''
    ### Plots
    #time = np.linspace(0,T,T+1)
    #plotter(time,DataAF/N,DataPF/N,'F.png')

    time = np.linspace(0,H,H+1)
    plotter(time,DataAH/N,DataPH/N,'results/dyanmics.png')

    return 

if __name__ == "__main__":
    config = yaml.load(open('Config-DataGen.yaml', 'r'), Loader=yaml.FullLoader)
    thetaP = list(np.loadtxt('results/seed3-rand-mu5.txt'))
    main(config, thetaP)   