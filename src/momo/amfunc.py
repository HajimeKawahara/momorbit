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
    mask=(Ea<np.pi)
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

def amf_relative(t,T0,P,e,omegaA,OmegaL,a,i,d):
    #relative astrometric orbit  
    X,Y,Z=XYZf(t,T0,P,e,omegaA,OmegaL,a,i)
    dRA=Y/d
    dDec=X/d
    return dRA, dDec
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from astropy.time import Time
    import sys
    JDYEAR=365.25 #Julian year
#    JDYEAR=365. #Conventional Year

    name="HIP11253"
#    name="GL22A"
    if name=="GL22A":
        P = 15.4275*JDYEAR #[day]
        T0=57447.0 + 2400000.5 #[JD]
        t=np.linspace(0,1,365)*P+T0        
        #prediction position
        pre = Time('2020-9-16 2:00:00').jd
        print(pre)
        print((pre-T0)/P)
        par=99.20 #parallax mas
        asec= 0.51026 #a [arcsec]
        a = asec/(par*1.e-3)        #[A.U.]
        d = 1.0/(par*1.e-3) #pc
        e = 0.1576
        i = 44.29       #[deg]
        
        node = 176.75    #[deg]
        w = 104.90      #[deg]    
        tp=57447.0 + 2400000.5
        
    elif name=="HIP11253":
        
        #
        P = 43.2*JDYEAR #[day]
        T0dec = Time(2026.5,format='decimalyear')
        T0=T0dec.jd #[JD]
        t=np.linspace(0,1,365)*P+T0
        pre = Time('2020-9-04 2:00:00').jd
            
        par=18.9878 #parallax mas
        asec= 0.271 #a [arcsec]
        a = asec/(par*1.e-3)        #[A.U.]
        d = 1.0/(par*1.e-3) #pc
        e = 0.89
        i = 133.5       #[deg]
        node = 214.8    #[deg]
        w = 110.3      #[deg]
    else:
        sys.exit("No name")
        
    i=i/180*np.pi
    omegaA = w*np.pi/180.0      #[rad]
    OmegaL = node*np.pi/180.0    #[rad]

    
    #X,Y,Z=XYZf(t,T0,P,e,omegaA,OmegaL,a,i)
    dRA,dDec=amf_relative(t,T0,P,e,omegaA,OmegaL,a,i,d)
    dRA_p,dDec_p=amf_relative(np.array([pre,T0]),T0,P,e,omegaA,OmegaL,a,i,d)
    print(dRA_p,dDec_p)

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    
    coord1 = SkyCoord(0*u.arcsec, 0*u.arcsec, frame='icrs')
    coord2 = SkyCoord(dRA_p[0]*u.arcsec, dDec_p[0]*u.arcsec, frame='icrs')
    pa_pre = coord1.position_angle(coord2).to(u.deg).value
    sep_pre = coord1.separation(coord2).to(u.arcsec)
    print(sep_pre)
    print(pa_pre)
    
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect=1.0)
    ax.plot(dRA,dDec)
    ax.plot(dRA_p,dDec_p,"s")
    ax.plot([0],[0],"+")
    plt.xlim(-a/d*(1+e),a/d*(1+e))
    plt.ylim(-a/d*(1+e),a/d*(1+e))
    plt.gca().invert_xaxis()
    plt.show()

