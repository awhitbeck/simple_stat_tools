import ROOT as r
from scipy.optimize import minimize
import numpy.random as rand
import numpy as np
import matplotlib.pyplot as plt

mu_inj=10.     # signal strength modifier for generating toys
s0=6.          # nominal signal expected
b0=0.1         # nominal background expected
sigma_b=0.1    # uncertainty for background
sigma_s=0.1    # uncertainty for signal

data = rand.poisson(b0+0.*s0,1)
n = float(len(data))

def nll(args):
    mu=args[0]
    s=args[1]
    b=args[2]
    arg = mu*s+b
    if arg <=  0 : return 999999999.
    res = 0.
    for d in data : 
        res = res-d
    return res*np.log(arg)+n*arg+((b-b0)**2)/2/(sigma_b*b0)**2+((s-s0)**2)/2/(sigma_s*s0)**2

def generate(args):
    return rand.poisson(args[0]*s0+b0,1)

result = minimize(nll,x0=(0.,s0,b0),method='BFGS',tol=1e-6)        
print 'result: a =',result.x
#print 'unc: ',result.hess_inv
min_nll = nll(result.x)

nlls=[]
mus = np.arange(0,3,0.01)
for mu in mus : 
    nlls.append(nll((mu,result.x[1],result.x[2]))-min_nll)
print zip(mus,nlls)
plt.plot(mus,nlls)
plt.show()

h = r.TH1F("test","test",int(s0+b0+4*np.sqrt(s0+b0)),0,int(s0+b0+4*np.sqrt(s0+b0)))
hs = r.TH1F("fitted_r",";r;Number of Toys",60,mu_inj-3.,mu_inj+3.)
pull = r.TH1F("pull",";(#hat{r}-r)/#sigma_{r};Number of Toys",60,-3.,3.)
hs.SetLineColor(2)

for i in range(10000):
    if i % 1000 == 0 : print 'i',i,'/',10000
    data = generate((mu_inj,0.,0.))
    h.Fill(data)
    fit_res = minimize(nll,x0=(0.,s0,b0),method='Nelder-Mead',tol=1e-6)
    hs.Fill(fit_res.x[0])

    mu=fit_res.x[0]
    #print 'best fit:',mu
    min_nll=nll((fit_res.x[0],fit_res.x[1],fit_res.x[2]))
    nll_mu=min_nll
    #print 'DetlaNLL',nll_mu-min_nll
    while (nll_mu-min_nll) < 0.5:
        mu+=0.01
        nll_mu=nll((mu,fit_res.x[1],fit_res.x[2]))
        #print 'DetlaNLL',nll_mu-min_nll,'mu',mu

        if mu>10. : break
    #print 'mu',mu,'nll',nll_mu,'nll( (mu-0.01,fit_res.x[1],fit_res.x[2]) ) )',nll( (mu-0.01,fit_res.x[1],fit_res.x[2]) ),'fit_res.x[0]',fit_res.x[0]
    #print '(nll_mu-min_nll-0.5)*0.01/( nll_mu-nll( (mu-0.01,fit_res.x[1],fit_res.x[2]) ) )',(nll_mu-min_nll-0.5)*0.01/( nll_mu-nll( (mu-0.01,fit_res.x[1],fit_res.x[2]) ) )
    mu_err=mu-(nll_mu-min_nll-0.5)*0.01/( nll_mu-nll( (mu-0.01,fit_res.x[1],fit_res.x[2]) ) )-fit_res.x[0]
    #print 'mu_err',mu_err
    pull.Fill((fit_res.x[0]-mu_inj)/mu_err)

can=r.TCanvas("can","can",500,500)
h.Draw()

can2=r.TCanvas("can2","can2",500,500)
hs.Draw()

can3=r.TCanvas("can3","can3",500,500)
pull.Draw()
