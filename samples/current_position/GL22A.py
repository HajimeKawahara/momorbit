if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from astropy.time import Time
    import numpy as np
    import sys
    from momo import amfunc
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    pre = Time('2020-9-16 2:00:00').jd
    
    JDYEAR=365.25 #Julian year
#    JDYEAR=365. #Conventional Year

    P = 15.4275*JDYEAR #[day]
    T0=57447.0 + 2400000.5 #[JD]
    t=np.linspace(0,1,365)*P+T0        

    #prediction position
    par=99.20 #parallax mas
    asec= 0.51026 #a [arcsec]
    a = asec/(par*1.e-3)        #[A.U.]
    d = 1.0/(par*1.e-3) #pc
    e = 0.1576
    i = 44.29       #[deg]        
    node = 176.75    #[deg]
    w = 104.90      #[deg]    
    tp=57447.0 + 2400000.5
        
    #deg to radian
    i=i/180*np.pi
    omegaA = w*np.pi/180.0      #[rad]
    OmegaL = node*np.pi/180.0    #[rad]

    
    #X,Y,Z=XYZf(t,T0,P,e,omegaA,OmegaL,a,i)
    dRA,dDec=amfunc.amf_relative(t,T0,P,e,omegaA,OmegaL,a,i,d)
    dRA_p,dDec_p=amfunc.amf_relative(np.array([pre,T0]),T0,P,e,omegaA,OmegaL,a,i,d)
    
    coord1 = SkyCoord(0*u.arcsec, 0*u.arcsec, frame='icrs')
    coord2 = SkyCoord(dRA_p[0]*u.arcsec, dDec_p[0]*u.arcsec, frame='icrs')
    pa_pre = coord1.position_angle(coord2).to(u.deg).value
    sep_pre = coord1.separation(coord2).to(u.arcsec)
    print(pre)
    print("Separation=",sep_pre,"arcsec")
    print("PA=",pa_pre,"degree")
    
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect=1.0)
    ax.plot(dRA,dDec)
    ax.plot(dRA_p,dDec_p,"s")
    ax.plot([0],[0],"+")
    plt.xlim(-a/d*(1+e),a/d*(1+e))
    plt.ylim(-a/d*(1+e),a/d*(1+e))
    plt.gca().invert_xaxis()
    plt.show()
