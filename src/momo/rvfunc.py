import numpy as np
import sys
from PyAstronomy import pyasl

def rvf(t,T0,P,e,omegaA,K,i,Vsys):
    ks = pyasl.MarkleyKESolver()
    n=2*np.pi/P
    M=n*(t-T0)

    Ea=[]
    for Meach in M:
        Eeach=ks.getE(Meach, e) #eccentric anomaly
        Ea.append(ks.getE(Meach, e))        
    Ea=np.array(Ea)
    
    cosE=np.cos(Ea)
    cosf=(-cosE + e)/(-1 + cosE*e)
    sinf=np.sqrt((-1 + cosE*cosE)*(-1 + e*e))/(-1 + cosE*e)
    mask=(Ea<np.pi)
    sinf[mask]=-sinf[mask]
    
    cosfpo=cosf*np.cos(omegaA)-sinf*np.sin(omegaA)
    face=1.0/np.sqrt(1.0-e*e)
    Ksini=K*np.sin(i)
    model = Vsys+Ksini*face*(cosfpo+e*np.cos(omegaA))
    
    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t=np.linspace(0,1.0,100)
    T0=0
    P=0.25
    e=0.85
    omegaA=np.pi
    K=3.0
    i=np.pi/2.0
    Vsys=1.0
    rv=rvf(t,T0,P,e,omegaA,K,i,Vsys)
    sys.exit()
    plt.plot(t,rv,".")
    plt.plot(t,rv)
    plt.show()
