from astropy.constants import G
from astropy.constants import M_sun
from astropy.constants import M_earth
from astropy import units as u
import numpy as np

#momoconstant
JDYEAR=365.25 #Julian year
GSYEAR=365.256898 #Gaussian year
ANORM=0.019570460672296595 #a normalized by P(d) and M (Msol). it can be checked by get_anorm()


def get_G_cuberoot():
    #  cuberoot of Gravitaional constant (km/s) normalized by day and Msun
    day=24*3600*u.s
    Gu=(G*M_sun/day).value
    Gcr=Gu**(1.0/3.0)*1.e-3
    return Gcr

def get_anorm():
    M=M_sun
    c=((G*M)/(4.0*np.pi**2)*(1*u.d)**2).to(u.au**3)
    anorm_check=(c**(1.0/3.0)).value
    return anorm_check

def PM2a(P,M):
    # provide a [AU] from P [d] and M [Msol] according to Kepler 3rd law
    return ANORM*(P**(2.0/3.0))*(M**(1.0/3.0))
    
if __name__ == "__main__":
    Gcr=get_G_cuberoot()
    print(Gcr)
    print("Earth velocity =",(2.0*np.pi)**(1.0/3.0)*Gcr*(365.25)**(-1.0/3.0),"km/s")
    print(PM2a(GSYEAR,1),"AU")
