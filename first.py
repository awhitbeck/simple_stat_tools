import numpy as np
from scipy.stats import poisson, lognorm, norm
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class naive_limit:

    def __init__(self, llh):
        self.llh = llh

    def integrate_likelihood(self, mu_min, mu_max, N=1000):
        # integrate the likelihood
        # return the integral value
        val = 0
        for i in range(N):
            val+=self.llh.eval(mu_min+i*(mu_max-mu_min)/N)*(mu_max-mu_min)/N
        return val

    def plot_likelihood(self, mu_min, mu_max, N=1000):
        # plot the likelihood
        # return the plot
        y = []
        for i in np.arange(mu_min,mu_max,(mu_max-mu_min)/N):
            y.append(self.llh.eval(i))
        plt.plot(np.arange(mu_min,mu_max,(mu_max-mu_min)/N),y)
        plt.show()
    def compute_limit(self, CL=0.95):
        # compute the limit
        # return the limit value
        temp=0

        norm = integrate.quad(self.llh.eval, 0, 50)[0]
        print('norm: ',norm)
        for i in np.arange(0,50,1):
            print("i: ",i," integral: ",integrate.quad(self.llh.eval, 0, i)[0]/norm)
            if integrate.quad(self.llh.eval, 0, i)[0] / norm > CL:
                temp = i
                break
        for i in np.arange(temp-1,temp+1,0.1):
            #print("i: ",i," integral: ",integrate.quad(self.llh.eval, 0, i)[0] / norm)
            if integrate.quad(self.llh.eval, 0, i)[0] / norm > CL:
                temp = i
                break
        for i in np.arange(temp-0.1,temp+0.1,0.01):
            #print("i: ",i," integral: ",integrate.quad(self.llh.eval, 0, i)[0] / norm)
            if integrate.quad(self.llh.eval, 0, i)[0] / norm > CL:
                return i
        print("No limit found")
        return -9999.

    def run_toy_experiments(self, mu, N=100):
        # run toy experiments
        # return the results
        results = []
        for i in range(N):
            self.llh.set_data(self.llh.generate_data(mu))
            results.append(self.compute_limit())
            print('i: ',i,' limit: ',results[i])
        return results

class multibin_likelihood:
    def __init__(self, num_bins):
        self.num_bins = num_bins
        self.bin = []
        self.bin_name = []
        self.mu = 0.
        self.nuis_type = []
        self.nuis_name = []
        self.nuis_unc = []
        self.nuis = []

    def add_systematic(self, bin_index, name, type, unc):
        if not name in self.nuis_name:
            self.nuis_name.append(name)
            self.nuis_type.append(type)
            self.nuis.append(1.0)
        self.bin(bin_index).add_nuisance(name, type, unc)

    def get_bin_index(self, name):
        if name in self.bin_name:
            return self.bin_name.index(name)
        else:
            return -1
    def add_bin(self, name = "", data_obs=0, bkg_exp=[], sig_exp=[]):
        if name in self.bin_name:
            print("ERROR: bin name already exists")
            return
        self.bin.append(likelihood(data_obs, bkg_exp, sig_exp))
        self.bin_name.append(name)

    def eval(self, mu):
        return np.prod([self.bin[i].eval(mu) for i in range(self.num_bins)])

    def generate_data(self, mu):
        return [self.bin[i].generate_data(mu) for i in range(self.num_bins)]

    def set_data(self, data):
        for i in range(self.num_bins):
            self.bin[i].set_data(data[i])
    def set_mu(self, mu):
            self.mu = mu
    def get_mu(self):
        return self.mu

class likelihood:
    def __init__(self, data_obs=0, bkg_exp=[], sig_exp=[]):
        self.data_obs = data_obs
        self.bkg_exp = bkg_exp
        self.sig_exp = sig_exp
        self.mu = 0.
        self.nuis_type = []
        self.nuis_name = []
        self.nuis_unc = []
        self.nuis = []

    def add_nuisance(self, name, type, unc):
        self.nuis_name.append(name)
        self.nuis_type.append(type)
        self.nuis_unc.append(unc)

    def eval_with_nuis(self,x=[],mu=0.):
        prior = np.prod([lognorm.pdf(value[0],value[1]-1.0) for name,value in self.nuis ])
        return self.eval(x[0],prior)

    #function that evaluates the log likelihood
    def eval(self,mu,nuis=[]):

        #print('lambda: ',lamb,' data: ',self.data_obs)
        #print('poisson: ',poisson.pmf(self.data_obs, lamb))
        if nuis == []:
            ## the signal nuisance must be first!
            lamb = sum(self.bkg_exp*nuis[1:]) + mu*sum(self.sig_exp*nuis[0])
            prior = np.prod([lognorm.logpdf(value,unc-1.0) for value,unc in zip(nuis,self.nuis_unc)])
            return -poisson.logpmf(self.data_obs, lamb)-prior

        else :
            prior = np.prod([lognorm.logpdf(value,unc-1.0) for value,unc in zip(self.nuis,self.nuis_unc)])
            return -poisson.logpmf(self.data_obs, lamb) - prior

    def generate_data(self, mu):
        lamb = sum(self.bkg_exp) + mu*sum(self.sig_exp)
        return np.random.poisson(lamb)

    def set_data(self, data_obs):
        self.data_obs = data_obs
    def set_mu(self, mu):
        self.mu = mu
    def get_mu(self):
        return self.mu

def test():
    # test the class
    # return the limit value
    data_obs = [10000,1000, 200, 100, 10]
    bkg_exp = [[10000],[1000],[200],[100],[10]]
    sig_exp = [[10],[10],[10],[10],[1]]
    l = multibin_likelihood(len(data_obs))
    for i in range(len(data_obs)):
        l.add_bin(data_obs[i], bkg_exp[i], sig_exp[i])
    lim = naive_limit(l)
    #print(l.integrate_likelihood(0, 1000, 1000))
    lim.plot_likelihood(0, 10, 1000)
    print("limits: ",lim.compute_limit())

    limits = lim.run_toy_experiments(0, 300)
    plt.hist(limits, bins=50)
    plt.show()

if __name__ == "__main__":
    test()