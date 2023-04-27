from scipy.optimize import minimize
from scipy.stats import poisson, lognorm, norm
from math import log
import numpy as np
import matplotlib.pyplot as plt
#from first import *
#from datacard_reader import *

def nll(nuis, b_unc, s_unc, mu, obs, b,s):
    b_nuis = nuis[0]
    s_nuis = nuis[1]
    #print('b_nuis: ',b_nuis,' s_nuis: ',s_nuis)
    #print('b_unc: ',b_unc,' s_unc: ',s_unc)
    #print('mu: ',mu,' obs: ',obs,' b: ',b)
    lam = b*b_nuis + mu*s*s_nuis
    #print('norm ',lognorm.logpdf(b_nuis,b_unc-1.0))
    #print('norm ',lognorm.logpdf(s_nuis,s_unc-1.0))
    #print('poisson ',poisson.pmf(obs, lam))
    return -poisson.logpmf(obs, lam)-lognorm.logpdf(b_nuis, b_unc-1.0)-lognorm.logpdf(s_nuis, s_unc-1.0)

b_unc = 1.10
s_unc = 1.10
mu = 0.5
obs = 120
b = 100
s = 10
x = minimize(nll, [1.,1.], args=(b_unc,s_unc,mu,obs,b,s),bounds = ((0, None), (0, None)))
print('minimum: ',x.x)
print('nll: ',nll(x.x, b_unc,s_unc,mu,obs,b,s))

x =[]
y =[]
minnll = 999999.
for i in np.arange(0,5,0.02):
    mu = i
    x.append(i)
    res = minimize(nll, [1.,1.], args=(b_unc,s_unc,mu,obs,b,s),bounds = ((0, None), (0, None)))
    temp_nll = nll(res.x, b_unc,s_unc,mu,obs,b,s)
    y.append(temp_nll)
    if temp_nll < minnll:
        minnll = temp_nll

y = np.subtract(y,minnll)
plt.plot(x,y),
plt.show()