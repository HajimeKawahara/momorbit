import numpy as np
from PyAstronomy import pyasl

def XYZf(t,T0,P,e,omegaA,OmegaL,a,i):
    #position
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
    mask=(Ea>=np.pi)
    sinf[mask]=-sinf[mask]
    cosoA=np.cos(omegaA)
    sinoA=np.sin(omegaA)
    cosOL=np.cos(OmegaL)
    sinOL=np.sin(OmegaL)
    
    cosfoA=cosf*cosoA - sinf*sinoA
    sinfoA=cosf*sinoA + sinf*cosoA
    
    r=a*(1.0 - e*e)/(1.0 + e*cosf)
    X=r*(cosfoA*cosOL - np.cos(i)*sinfoA*sinOL)
    Y=r*(np.cos(i)*cosOL*sinfoA + cosfoA*sinOL)
    Z=r*(np.sin(i)*sinfoA)

    return X,Y,Z

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t=np.linspace(0,1.0,100)
    T0=0
    P=0.25
    e=0.3
    omegaA=np.pi
    OmegaL=np.pi
    a=1.0
    i=np.pi/3.0

    X,Y,Z=XYZf(t,T0,P,e,omegaA,OmegaL,a,i)
    
    plt.plot(X,Y,".")
    plt.show()

