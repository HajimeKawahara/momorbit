import numpy as np
import sys
from PyAstronomy import pyasl
from momo import momoconst

Gcr=momoconst.get_G_cuberoot()
fac=(2.0*np.pi)**(1.0/3.0)
m23=-2.0/3.0
m13=-1.0/3.0

def rvf2(t,T0,P,e,omegaA,M1,M2,i,Vsys):
    #RV of M1
    K=fac*Gcr*M2*((M1+M2)**m23)*(P**m13)/np.sqrt(1.0 - e*e)
    return rvcoref(t,T0,P,e,omegaA,K,i) + Vsys

def rvf2c(t,T0,P,e,omegaA,M1,M2,i,Vsys):
    #RV of M2 (companion)
    K=fac*Gcr*M1*((M1+M2)**m23)*(P**m13)/np.sqrt(1.0 - e*e)
    return -rvcoref(t,T0,P,e,omegaA,K,i) + Vsys

def rvf1(t,T0,P,e,omegaA,K,i,Vsys):
    return rvcoref(t,T0,P,e,omegaA,K,i) + Vsys


def rvcoref(t,T0,P,e,omegaA,K,i):
    # semi amplitude is defined by K*sin(i)
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
    model = Ksini*face*(cosfpo+e*np.cos(omegaA))

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
