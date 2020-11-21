import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import corner
import emcee
from momo import rvfunc
from momo import amfunc
from momo import momoconst
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

#REACH value
tR0=Time('2020-9-4').jd
trvA=np.array([tR0])
trvB=np.array([tR0])
rvA=np.array([5.21])
rvB=np.array([-1.50])
e_rvA=np.array([0.1])
e_rvB=np.array([0.1])

#GAIA DR2 DISTANCE
d_in=1000.0/18.9878 #pc
sigma_d=d_in*0.6268/18.9878
inv_sigma2_d=1.0/sigma_d/sigma_d

#Setting a probability model
def lnprob(p, trv, rv, e_rv, x, y, asterr):
    T0,P,e,omegaA,MA,MB,Vsys,OmegaL,d,i,sigunk_rv,sigunk_ast = p
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    lnp = lp + lnlike_rvA(T0,P,e,omegaA,MA,MB,i,Vsys,trv,rv,e_rv,sigunk_rv) \
          + lnlike_rvA(T0,P,e,omegaA,MA,MB,i,Vsys,trvA,rvA,e_rvA,0.0) \
          + lnlike_rvB(T0,P,e,omegaA,MA,MB,i,Vsys,trvB,rvB,e_rvB,0.0) \
          + lnlike_ast(T0,P,e,omegaA,OmegaL,MA,MB,d,i,tast,x,y,asterr,sigunk_ast) 
    return lnp

def lnlike_rvA(T0,P,e,omegaA,M1,M2,i,Vsys,trv,rv,e_rv,sigunk_rv):
    rvmodel=rvfunc.rvf2(trv,T0,P,e,omegaA,M1,M2,i,Vsys)    
    inv_sigma2 = 1.0/(e_rv**2 + sigunk_rv**2)
    lnRV=-0.5*(np.sum((rv-rvmodel)**2*inv_sigma2 - np.log(inv_sigma2)))    
    return lnRV 

def lnlike_rvB(T0,P,e,omegaA,M1,M2,i,Vsys,trv,rv,e_rv,sigunk_rv):
    rvmodel=rvfunc.rvf2c(trv,T0,P,e,omegaA,M1,M2,i,Vsys)    
    inv_sigma2 = 1.0/(e_rv**2 + sigunk_rv**2)
    lnRV=-0.5*(np.sum((rv-rvmodel)**2*inv_sigma2 - np.log(inv_sigma2)))    
    return lnRV 


def lnlike_ast(T0,P,e,omegaA,OmegaL,M1,M2,d,i,tast,x,y,asterr,sigunk_ast):
    dra_model,ddec_model=amfunc.amf_relative2(tast,T0,P,e,omegaA,OmegaL,M1,M2,d,i)    
    inv_sigma2 = 1.0/(asterr**2 + sigunk_ast**2)
    lnAST=-0.5*(np.sum(((x-dra_model)**2 + (y-ddec_model)**2)*inv_sigma2 - np.log(inv_sigma2)))    
    return lnAST

def lnprior(p):
    T0,P,e,omegaA,M1,M2,Vsys,OmegaL,d,i,sigunk_rv,sigunk_ast = p
    if 0.0 <= e < 1.0 and 0.0 <= i <= 2.0*np.pi and 0.0 <= omegaA < 2.0*np.pi \
       and 0.0 <= OmegaL < 2.0*np.pi and 0.0 <= M1 and 0.0 <= M2 and 0.0 <= d\
       and 0.0 <= sigunk_rv and 0.0 <= sigunk_ast:    
        return -0.5*(d_in - d)**2*inv_sigma2_d - np.log(inv_sigma2_d)
    
    return -np.inf

#Initial Values
JDYEAR=365.25 #Julian year
P_in = (43.2)*JDYEAR #[day]
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
Vsys_in=2.25
MA_in = 1.0
MB_in = 0.7
sigunk_rv_in=np.mean(e_rv)/10.0
sigunk_ast_in=np.mean(asterr)/10.0

#Checking initial fit
trvw=np.linspace(trv[0]-P_in/2,trv[-1]+P_in/2,1000)
rvmodelA=rvfunc.rvf2(trvw,T0_in,P_in,e_in,omegaA_in,MA_in,MB_in,i_in,Vsys_in)
rvmodelB=rvfunc.rvf2c(trvw,T0_in,P_in,e_in,omegaA_in,MA_in,MB_in,i_in,Vsys_in)

tastw=np.linspace(tast[0],P_in+tast[0],1000)
dra_model,ddec_model=amfunc.amf_relative2(tastw,T0_in,P_in,e_in,omegaA_in,OmegaL_in,MA_in,MB_in,d_in,i_in)


fig=plt.figure(figsize=(25,7))
ax=fig.add_subplot(131)
ax.plot(trvw,rvmodelA,c="C0")
ax.plot(trvw,rvmodelB,c="C1")

ax.plot(trv,rv,"*",c="C0")
ax.plot(trvA,rvA,"^",c="C0")
ax.plot(trvB,rvB,"s",c="C1")

plt.axhline(Vsys_in,ls="dashed")
ax=fig.add_subplot(132)
ax.plot(dra_model,ddec_model)
ax.plot(x,y,"*")
plt.gca().invert_xaxis()

ax=fig.add_subplot(133)
ax.plot(tastw,dra_model,c="C0")
ax.plot(tast,x,"*",c="C0")
ax.plot(tastw,ddec_model,c="C1")
ax.plot(tast,y,"*",c="C1")

plt.show()
#------------------------------
pin=np.array([T0_in,P_in,e_in,omegaA_in,MA_in,MB_in,Vsys_in,OmegaL_in,d_in,i_in,sigunk_rv_in,sigunk_ast_in])
nwalkers = 100
ndim=len(pin)
err=np.array([10.0,10.0,0.01,np.pi/1000,0.3,0.2,0.3,np.pi/1000,sigma_d/10.0,np.pi/1000,sigunk_rv_in*1.e-2,sigunk_ast_in*1.e-2])
pos = [pin + err*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(trv, rv, e_rv, x, y, asterr))
sampler.run_mcmc(pos, 5000,progress=True);

samples = sampler.get_chain(discard=500,thin=15,flat=True)

np.savez("sample_SB2AST.npz",samples)

fig=plt.figure()
plt.plot(samples[:,2],".")
plt.savefig("e.png")

fig=plt.figure(figsize=(25,7))
ax=fig.add_subplot(121)
ax.plot(trv,rv,"*",c="C0",label="Tokovinin+ A")
ax.plot(trvA,rvA,"^",c="blue",label="REACH A")
ax.plot(trvB,rvB,"s",c="red",label="REACH B")
ax.legend()

ax2=fig.add_subplot(122)
ax2.plot(x,y,"*")
plt.gca().invert_xaxis()

#prediction
trvw=np.linspace(trv[0]-P_in/2,trv[-1]+P_in/2,1000)
tastw=np.linspace(0,P_in,1000)
inds= np.random.randint(len(samples), size=100)
for ind in inds:
    samp=samples[ind]
    T0,P,e,omegaA,MA,MB,Vsys,OmegaL,d,i,sigunk_rv,sigunk_ast = samp
    rvmodelA=rvfunc.rvf2(trvw,T0,P,e,omegaA,MA,MB,i,Vsys)    
    rvmodelB=rvfunc.rvf2c(trvw,T0,P,e,omegaA,MA,MB,i,Vsys)    
    dra_model,ddec_model=amfunc.amf_relative2(tastw,T0,P,e,omegaA,OmegaL,MA,MB,d,i)    
    ax.plot(trvw,rvmodelA,alpha=0.02,c="green")
    ax.plot(trvw,rvmodelB,alpha=0.02,c="magenta")
    ax2.plot(dra_model,ddec_model,alpha=0.05,c="green")
plt.savefig("pred_SB2AST.png")


#mass corner
fig = corner.corner(samples[:,4:6], labels=["$M_A$","M_B"])
plt.savefig("corner_mass_SB2AST.png")

#corner
labp=np.array(["T0","P","e","$\omega$","$M_A$", "$M_B$","$V_{sys}$","$\Omega$","a","i","$\sigma_r$","$\sigma_a$"])
fig = corner.corner(samples, labels=labp,truths=pin)
plt.savefig("corner_all_SB2AST.png")
