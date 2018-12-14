import abc
import enum
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal
from scipy.stats.mstats import gmean


class PricingEngine(object, metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def calculate(self):
        """A method to implement a pricing model.
           The pricing method may be either an analytic model (i.e.
           Black-Scholes), a PDF solver such as the finite difference method,
           or a Monte Carlo pricing algorithm.
        """
        pass
    
class MonteCarloEngine(PricingEngine):
    def __init__(self, replications, time_steps, pricer):
        self.__replications = replications
        self.__time_steps = time_steps
        self.__pricer = pricer

    @property
    def replications(self):
        return self.__replications

    @replications.setter
    def replications(self, new_replications):
        self.__replications = new_replications

    @property
    def time_steps(self):
        return self.__time_steps

    @time_steps.setter
    def time_steps(self, new_time_steps):
        self.__time_steps = new_time_steps
    
    def calculate(self, option, data):
        return self.__pricer(self, option, data)
    
    
def NaiveMonteCarloPricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, vol, div) = data.get_data()
    replications = engine.replications
    dt = expiry / engine.time_steps
    disc = np.exp(-rate * dt)
    
    z = np.random.normal(size = replications)
    spotT = spot * np.exp((rate - div - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z)
    payoffT = option.payoff(spotT)

    prc = payoffT.mean() * disc

    return prc

def AssetPaths(spot, mu, vol, expiry, div, nreps, nsteps):
    paths = np.empty((nreps, nsteps + 1))
    h = expiry / nsteps
    paths[:, 0] = spot
    mudt = (mu - div - 0.5 * vol * vol) * h
    voldt = vol * np.sqrt(h)
    
    for t in range(1, nsteps + 1):
        z = np.random.normal(size=nreps)
        paths[:, t] = paths[:, t-1] * np.exp(mudt + voldt * z)

    return paths

def PathwiseNaiveMonteCarloPricer(engine, option, data):
    (spot, rate, vol, div) = data.get_data()
    expiry = option.expiry
    nreps = engine.replications
    nsteps = engine.time_steps
    paths = AssetPaths(spot, mu, vol, expiry, div, nreps, nsteps)
    call_t = 0
    
    for i in range(nreps):
        call_t += option.payoff(paths[i])
        
    call_t /= nreps
    call_t *= np.exp(-rate * expiry)
    
    return call_t

def AntitheticMonteCarloPricer(engine, option, data):
    expiry = option.expiry
    strike = option.strike
    (spot, rate, vol, div) = data.get_data()
    replications = engine.replications
    dt = expiry / engine.time_steps
    disc = np.exp(-(rate - div) * dt)
    
    z1 = np.random.normal(size = replications)
    z2 = -z1
    z = np.concatenate((z1,z2))
    spotT = spot * np.exp((rate - div) * dt + vol * np.sqrt(dt) * z)
    payoffT = option.payoff(spotT)

    prc = payoffT.mean() * disc

    return prc

def blackScholesCall(spot, strike, rate, vol, div, expiry):
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * np.sqrt(expiry))
    d2 = d1 - vol * np.sqrt(expiry)
    callPrice = (spot * np.exp(-div * expiry) * norm.cdf(d1)) - (strike * np.exp(-rate * expiry)  * norm.cdf(d2))
    return callPrice

def geometricAsianCall(spot, strike, rate, vol, div, expiry, N):
    dt = expiry / N
    nu = rate - div - 0.5 * vol * vol
    a = N * (N+1) * (2.0 * N + 1.0) / 6.0
    V = np.exp(-rate * expiry) * spot * np.exp(((N + 1.0) * nu / 2.0 + vol * vol * a / (2.0 * N * N)) * dt)
    vavg = vol * np.sqrt(a) / pow(N, 1.5)
    callPrice = blackScholesCall(V, strike, rate, vavg, div, expiry)
    return callPrice

def controlVariateAsianCall(engine, option, data):
    T = option.expiry
    K = option.strike
    (spot, rate, vol, div) = data.get_data()
    M = engine.replications
    N = engine.time_steps
    dt = T / N
    paths = AssetPaths(spot, rate, vol, T, div, M, N)
    nudt = (rate - div - 0.5 *  vol ** 2) * dt
    sigstd = vol * (dt) ** 0.5
    sum_CT = 0
    sum_CT2 = 0    
    for j in range(M):
        A = np.mean(paths[j])
        G = gmean(paths[j])
        CT = option.payoff(A) - option.payoff(G)
        sum_CT += CT
        sum_CT2 += CT * CT
    portfolio_value = sum_CT / M * np.exp(-rate * T)
    SD = ((sum_CT2 - sum_CT * sum_CT / M) * np.exp(-2 * rate * T) / (M - 1)) ** 0.5
    SE = SD / ((M) ** 0.5)
    price = portfolio_value + geometricAsianCall(spot, K, rate, vol, div, T, N)
    return (price, SE)

def simpleMonteCarloAsianCall(engine, option, data):
    T = option.expiry
    K = option.strike
    (spot, rate, vol, div) = data.get_data()
    M = engine.replications
    N = engine.time_steps
    dt = T / N
    paths = AssetPaths(spot, rate, vol, T, div, M, N)
    nudt = (rate - div - 0.5 *  vol ** 2) * dt
    sigstd = vol * (dt) ** 0.5
    sum_CT = 0
    sum_CT2 = 0    
    for j in range(M):
        A = np.mean(paths[j])
        CT = option.payoff(A)
        sum_CT += CT
        sum_CT2 += CT * CT
    portfolio_value = sum_CT / M * np.exp(-rate * T)
    SD = ((sum_CT2 - sum_CT * sum_CT / M) * np.exp(-2 * rate * T) / (M - 1)) ** 0.5
    SE = SD / ((M) ** 0.5)
    price = portfolio_value
    return (price, SE)