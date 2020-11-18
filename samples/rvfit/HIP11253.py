import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import emcee
from momo import rvfunc
from momo import amfunc


datrv=pd.read_csv("../../data/HIP11253RV.txt",delimiter=",",comment="#")
trv=datrv["Date"]+2400000 #JD
rv=datrv["Rvel"]
e_rv=datrv["e_Rvel"]

datast=pd.read_csv("../../data/HIP11253AST.txt",delimiter=",",comment="#")
tastDY=datast["Date"]
tast = Time(tastDY,format='decimalyear').jd #JD
pa=datast["PA"]/180*np.pi
sep=datast["sep"]
x=sep*np.sin(pa)
y=sep*np.cos(pa)

asterr=datast["e_sep"]

#rvfunc.rvf(t,T0,P,e,omega,Ksini,Vsys)
#amfunc.amf_relative(t,T0,P,e,omegaA,OmegaL,a,i,d)

def lnprob(p, trv, rv, e_rv, x, y, asterr):
    T0,P,e,omegaA,Ksini,Vsys,OmegaL,a,i,sigunk_rv,sigunk_ast = p

    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    lnp = lp \
        + lnlike_rv(T0,P,e,omegaA,Ksini,Vsys,trv,rv,e_rv,sigunk_rv) \
        + lnlike_ast(T0,P,e,omegaA,OmegaL,a,i,tast,x,y,asterr,sigunk_ast)
    return lnp

def lnlike_rv(T0,P,e,omegaA,Ksini,Vsys,trv,rv,e_rv,sigunk_rv):
    rvmodel=rvfunc.rvf(trv,T0,P,e,omegaA,Ksini,Vsys)    
    inv_sigma2 = 1.0/(e_rv**2 + sigunk_rv**2)
    lnRV=-0.5*(np.sum((rv-rvmodel)**2*inv_sigma2 - np.log(inv_sigma2)))    
    return lnRV 

def lnlike_ast(T0,P,e,omegaA,OmegaL,a,i,tast,x,y,asterr,sigunk_ast):
    dramodel,ddecmodel=amfunc.amf_relative_direct(t,T0,P,e,omegaA,OmegaL,a,i)
    inv_sigma2 = 1.0/(asterr**2 + sigunk_ast**2)
    lnAST=-0.5*(np.sum(((x-dra_model)**2 + (y-ddec_model)**2)*inv_sigma2 - np.log(inv_sigma2)))    
    return lnAST


def lnprior(p):
    T0,P,e,omegaA,Ksini,Vsys,OmegaL,a,i,sigunk_rv,sigunk_ast = p
    if 0.0 <= e < 1.0 and 0.0 <= i < 1.0 and 0.0 <= omegaA < 2.0*np.pi \
       and 0.0 <= OmegaL < 2.0*np.pi and 0.0 <= Ksini and 0.0 <= a\
       and 0.0 <= sigunk_rv and 0.0 <= sigunk_ast:    
        return 0.0
    
    return -np.inf

JDYEAR=365.25 #Julian year
#    JDYEAR=365. #Conventional Year

P_in = 15.4275*JDYEAR #[day]
T0_in=57447.0 + 2400000.5 #[JD]
a_in= 0.51026 #a [arcsec]
e_in = 0.1576
i_in = 44.29/180.0*np.pi
OmegaL_in = 176.75/180.0*np.pi
omegaA_in = 104.90/180.0*np.pi
sigunk_rv_in=0.1
sigunk_ast_in=0.1
Ksini_in=6.3
Vsys_in=2.5

pin=np.array([T0_in,P_in,e_in,omegaA_in,Ksini_in,Vsys_in,OmegaL_in,a_in,i_in,sigunk_rv_in,sigunk_ast_in])
nwalkers = 100
ndim=len(pin)
pos = [pin + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(trv, rv, e_rv, x, y, asterr))
sampler.run_mcmc(pos, 10000);

sys.exit()
plt.plot(x,y,".")
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.gca().invert_xaxis()
#plt.plot(trv,rv,".")
plt.show()


#    Ksini = np.sin(i)*(2.0*np.pi)**(1.0/3.0)*Mp
