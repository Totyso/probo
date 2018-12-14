## Tyson Clark and Jesse Baker

from probo.marketdata import MarketData
from probo.payoff import VanillaPayoff, call_payoff, put_payoff
from probo.engine import MonteCarloEngine, controlVariateAsianCall, simpleMonteCarloAsianCall
from probo.facade import OptionFacade

## Set up the market data
spot = 100.0
rate = 0.06
volatility = 0.20
dividend = 0.03
thedata = MarketData(rate, spot, volatility, dividend)

## Set up the option
expiry = 1.0
strike = 100.0
thecall = VanillaPayoff(expiry, strike, call_payoff)
theput = VanillaPayoff(expiry, strike, put_payoff)

## Set up controlVariateAsianCall
nreps = 10000
steps = 10
pricer = controlVariateAsianCall
mcengine = MonteCarloEngine(nreps, steps, pricer)
pricer2 = simpleMonteCarloAsianCall
mcengine1 = MonteCarloEngine(nreps, steps, pricer2)

## Calculate the price
option1 = OptionFacade(thecall, mcengine, thedata)
price1, se1 = option1.price()
print("The call price via controlVariateAsianCall is: {0:.3f}".format(price1))
print("The standard error via controlVariateAsianCall is: {0:.6f}".format(se1))
option2 = OptionFacade(thecall, mcengine1, thedata)
price2, se2 = option2.price()
print("The call price via simpleMonteCarloAsianCal is: {0:.3f}".format(price2))
print("The standard error via simpleMonteCarloAsianCal is: {0:.6f}".format(se2))
