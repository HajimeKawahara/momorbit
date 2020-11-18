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
    t=np.linspace(0,100.0,100)
    T0=0

    par=99.20 #parallax mas
    asec= 0.51026 #a [arcsec]
    a = asec/(par*1.e-3)        #[A.U.]
    e = 0.1576
    i = 44.29       #[deg]
    
    node = 176.75    #[deg]
    
    w = 104.90      #[deg]
    
    tp=57447.0 + 2400000.5
    P = 15.4275 #[yr]

    #
    par=18.9878 #parallax mas
    asec= 0.271 #a [arcsec]
    a = asec/(par*1.e-3)        #[A.U.]
    e = 0.89
    i = 133.5       #[deg]
    node = 214.8    #[deg]
    w = 110.3      #[deg]

    i=i/180*np.pi
    omegaA = w*np.pi/180.0      #[rad]
    OmegaL = node*np.pi/180.0    #[rad]

    
    X,Y,Z=XYZf(t,T0,P,e,omegaA,OmegaL,a,i)
    
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect=1.0)
    ax.plot(-Y,X,".")
    ax.plot([0],[0],"+")
    plt.xlim(-a*(1+e),a*(1+e))
    plt.ylim(-a*(1+e),a*(1+e))
    plt.show()

