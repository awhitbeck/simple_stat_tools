import numpy as np
from scipy.stats import poisson
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

    def set_data(self, data):
        for i in range(self.bin.num_bins):
            self.bin[i].set_data(data[i])
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
        #print('norm: ',norm)
        for i in np.arange(0,50,1):
            #print("i: ",i," integral: ",integrate.quad(self.eval, 0, i)[0]/norm)
            if integrate.quad(self.llh.eval, 0, i)[0] / norm > CL:
                temp = i
                break
        for i in np.arange(temp-1,temp+1,0.1):
            #print("i: ",i," integral: ",integrate.quad(self.eval, 0, i)[0] / norm)
            if integrate.quad(self.llh.eval, 0, i)[0] / norm > CL:
                temp = i
                break
        for i in np.arange(temp-0.1,temp+0.1,0.01):
            #print("i: ",i," integral: ",integrate.quad(self.eval, 0, i)[0] / norm)
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
        self.bins = []
        self.mu = 0.

    def add_bin(self, data_obs=0, bkg_exp=[], sig_exp=[]):
        self.bins.append(likelihood(data_obs, bkg_exp, sig_exp))

    def eval(self, mu):
        return np.prod([self.bins[i].eval(mu) for i in range(self.num_bins)])

    def generate_data(self, mu):
        return [self.bins[i].generate_data(mu) for i in range(self.num_bins)]

    def set_data(self, data):
        for i in range(self.num_bins):
            self.bins[i].set_data(data[i])
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
    def eval(self,mu):
        lamb = sum(self.bkg_exp) + mu*sum(self.sig_exp)
        return poisson.pmf(self.data_obs, lamb)

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

    limits = lim.run_toy_experiments(0, 100)
    plt.hist(limits, bins=100)
    plt.show()

if __name__ == "__main__":
    test()