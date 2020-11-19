import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import corner
import emcee
from momo import rvfunc
from momo import amfunc
import sys

#data import
datrv=pd.read_csv("../../data/HIP11253RV.txt",delimiter=",",comment="#")
trv=(datrv["Date"].values+2400000) #JD
rv=datrv["Rvel"].values
e_rv=datrv["e_Rvel"].values

datast=pd.read_csv("../../data/HIP11253AST.txt",delimiter=",",comment="#")
tastDY=datast["Date"]
tast = Time(tastDY,format='decimalyear').jd #JD
pa=datast["PA"].values/180*np.pi
sep=datast["sep"].values
x=sep*np.sin(pa)
y=sep*np.cos(pa)
asterr=datast["e_sep"].values

#Setting a probability model
def lnprob(p, trv, rv, e_rv, x, y, asterr):
    T0,P,e,omegaA,K,Vsys,OmegaL,a,i,sigunk_rv,sigunk_ast = p
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    lnp = lp \
        + lnlike_rv(T0,P,e,omegaA,K,i,Vsys,trv,rv,e_rv,sigunk_rv) \
        + lnlike_ast(T0,P,e,omegaA,OmegaL,a,i,tast,x,y,asterr,sigunk_ast)
    return lnp

def lnlike_rv(T0,P,e,omegaA,K,i,Vsys,trv,rv,e_rv,sigunk_rv):
    rvmodel=rvfunc.rvf(trv,T0,P,e,omegaA,K,i,Vsys)    
    inv_sigma2 = 1.0/(e_rv**2 + sigunk_rv**2)
    lnRV=-0.5*(np.sum((rv-rvmodel)**2*inv_sigma2 - np.log(inv_sigma2)))    
    return lnRV 

def lnlike_ast(T0,P,e,omegaA,OmegaL,a,i,tast,x,y,asterr,sigunk_ast):
    dra_model,ddec_model=amfunc.amf_relative_direct(tast,T0,P,e,omegaA,OmegaL,a,i)
    inv_sigma2 = 1.0/(asterr**2 + sigunk_ast**2)
    lnAST=-0.5*(np.sum(((x-dra_model)**2 + (y-ddec_model)**2)*inv_sigma2 - np.log(inv_sigma2)))    
    return lnAST

def lnprior(p):
    T0,P,e,omegaA,K,Vsys,OmegaL,a,i,sigunk_rv,sigunk_ast = p
    if 0.0 <= e < 1.0 and 0.0 <= i < 1.0 and 0.0 <= omegaA < 2.0*np.pi \
       and 0.0 <= OmegaL < 2.0*np.pi and 0.0 <= K and 0.0 <= a\
       and 0.0 <= sigunk_rv and 0.0 <= sigunk_ast:    
        return 0.0
    
    return -np.inf

#Initial Values
JDYEAR=365.25 #Julian year
P_in = 43.2*JDYEAR #[day]
T0dec = Time(2026.5,format='decimalyear')
T0_in=T0dec.jd #[JD]
#pre = Time('2020-9-04 2:00:00').jd
a_in= 0.271 #a [arcsec]
e_in = 0.89
i_in = 133.5       #[deg]
node = 214.8    #[deg]
w = 110.3      #[deg]

i_in=i_in/180*np.pi
omegaA_in = w*np.pi/180.0      #[rad]
OmegaL_in = node*np.pi/180.0    #[rad]
K_in=6.29
Vsys_in=2.25
sigunk_rv_in=np.mean(e_rv)/10.0
sigunk_ast_in=np.mean(asterr)/10.0

#Checking initial fit
trvw=np.linspace(trv[0]-P_in/2,trv[-1]+P_in/2,1000)
rvmodel=rvfunc.rvf(trvw,T0_in,P_in,e_in,omegaA_in,K_in,i_in,Vsys_in)    

tastw=np.linspace(0,P_in,1000)
dra_model,ddec_model=amfunc.amf_relative_direct(tastw,T0_in,P_in,e_in,omegaA_in,OmegaL_in,a_in,i_in)


fig=plt.figure(figsize=(25,7))
ax=fig.add_subplot(121)
ax.plot(trvw,rvmodel)
ax.plot(trv,rv,"*")
plt.axhline(Vsys_in,ls="dashed")
ax=fig.add_subplot(122)
ax.plot(dra_model,ddec_model)
ax.plot(x,y,"*")
plt.gca().invert_xaxis()
plt.show()
#------------------------------
pin=np.array([T0_in,P_in,e_in,omegaA_in,K_in,Vsys_in,OmegaL_in,a_in,i_in,sigunk_rv_in,sigunk_ast_in])
nwalkers = 300
ndim=len(pin)
err=np.array([10.0,10.0,0.01,np.pi/1000,0.3,0.3,np.pi/1000,0.01,np.pi/1000,sigunk_rv_in*1.e-2,sigunk_ast_in*1.e-2])
pos = [pin + err*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(trv, rv, e_rv, x, y, asterr))
sampler.run_mcmc(pos, 50000,progress=True);

samples = sampler.get_chain(discard=100,thin=15,flat=True)

fig=plt.figure(figsize=(25,7))
ax=fig.add_subplot(121)
ax.plot(trvw,rvmodel)
ax.plot(trv,rv,"*")
plt.axhline(Vsys_in,ls="dashed")

ax2=fig.add_subplot(122)
ax2.plot(dra_model,ddec_model)
ax2.plot(x,y,"*")
plt.gca().invert_xaxis()

trvw=np.linspace(trv[0]-P_in/2,trv[-1]+P_in/2,1000)
tastw=np.linspace(0,P_in,1000)
inds= np.random.randint(len(samples), size=100)
for ind in inds:
    samp=samples[ind]
    T0,P,e,omegaA,K,Vsys,OmegaL,a,i,sigunk_rv,sigunk_ast = samp
    rvmodel=rvfunc.rvf(trvw,T0,P,e,omegaA,K,i,Vsys)    
    dra_model,ddec_model=amfunc.amf_relative_direct(tastw,T0,P,e,omegaA,OmegaL,a,i)

    ax.plot(trvw,rvmodel,alpha=0.05,c="green")
    ax2.plot(dra_model,ddec_model,alpha=0.05,c="green")

plt.show()


#corner
labp=np.array(["T0","P","e","$\omega$","K","$V_{sys}$","$\Omega$","a","i","$\sigma_r$","$\sigma_a$"])
fig = corner.corner(samples, labels=labp,
                      truths=pin)
plt.savefig("corner.png")


#    Ksini = np.sin(i)*(2.0*np.pi)**(1.0/3.0)*Mp
