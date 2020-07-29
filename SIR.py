
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.linear_model import LinearRegression
from lmfit import Minimizer, Parameters, report_fit


import matplotlib.pyplot as plt

from scipy import interpolate



##########################
#### Contact rate profiles
##########################

def smooth_step(x, params):
# t<t1    => beta0 const
# t>t2    => beta2 const
# t1<t<t2 => cubic to be continuous and differentiable at t1 and t2
    beta0 = params['beta0']
    beta2 = params['beta2']
    t1    = params['t1']
    t2    = params['t2']

    b = beta0 * np.ones_like(x)   #t<t1

    s = (x-t1)/(t2-t1)   #s is in [0,1]
    b = np.where( (t1<=x)&(x<=t2), beta2 + (beta0-beta2)*(2*s**3 - 3*s**2 +1), b)   # y = 2.s^3 -3.s^2 + 1 has the C1 property we want for the result at t1 and t2

    b = np.where( (t2<x), beta2, b)
    
    return b

#x = np.arange(0,100)
#y = np.zeros(100)
#y = smooth_step(x, {'beta0':4/7, 'beta2':2/7, 't1': 20, 't2': 60})
#plt.plot(x, y,label='interv')
#plt.grid()


def reopening(x, params):  
#this function assumes a 'smooth-step' intervention (between t1 and t2), followed later by a linear increase (starting at t3 and continuing until t4)
    beta2 = params['beta2']  #no need for beta3, as function is constant at beta2 between t2 and t3
    beta4 = params['beta4']  #no need for beta3, as function is constant at beta2 between t2 and t3
    t3    = params['t3']
    t4    = params['t4']
    b = smooth_step(x, params)
    b = np.where( t3<=x, (beta4-beta2)/(t4-t3)*(x-t3)+beta2, b)
    return b


def piecewiselin(x, params):  
    #r  = params[0]
    b0 = params['beta0']
    n  = params['segments'] #number of segments
    init_beta = params['init_beta']

    bi1 = b0
    ti1 = x[0]

    if n==0:
        b = b0 * np.ones(len(x))
    else:
        b = np.zeros(len(x))
        i=1
        while i<=n:

            bi = params['beta{}'.format(i)]
            ti = params['t{}'.format(i)]
            
            if i==1 and init_beta == 'const':
                b = np.where( (ti1<=x)&(x<=ti), bi1, b)
            else:
                b = np.where( (ti1<=x)&(x<=ti), (x-ti1)*(bi-bi1)/(ti-ti1)+bi1, b)
            i=i+1
            bi1=bi
            ti1=ti

        b = np.where(x>=ti,bi,b)
    
    return b

#x = np.arange(100)
#b0 = piecewiselin(x, {'init_beta':'', 'segments':0, 'beta0':3/7})
#b1 = piecewiselin(x, {'init_beta':'', 'segments':1, 't1':20, 'beta0':3.1/7, 'beta1':2.1/7})
#b2 = piecewiselin(x, {'init_beta':'const', 'segments':2, 't1':20, 't2':80, 'beta0':3.2/7, 'beta1':2.2/7, 'beta2':1/7})
#b2l = piecewiselin(x, {'init_beta':'', 'segments':2, 't1':20, 't2':80, 'beta0':3.2/7, 'beta1':2.2/7, 'beta2':1/7})
#plt.plot(x,b0,label='b0')
#plt.plot(x,b0,label='b0')
#plt.plot(x,b1,label='b1')
#plt.plot(x,b2,label='b2')
#plt.plot(x,b2l,label='b2l')
#plt.grid()
#plt.legend()
#plt.show()


def piecewiseconst(x, params):  
    b0 = params['beta0']
    n  = params['segments'] #number of segments


    b = b0 * np.ones(len(x))
    i=1
    while i<=n:
        
        bi = params['beta{}'.format(i)]
        ti = params['t{}'.format(i)]
        b = np.where( x>=ti, bi, b)
        i=i+1
        
    return b




#-------------------------------------------------------
def contact_rate(x,params):
#time varying contact rate, using one of the available profiles
#calculate the contact rate over time, according to the profile given by the intervention function and the calibration params (betai, ti)

    interv_functions = {
                        'smooth step'        : smooth_step, 
                        'piecewise linear'   : piecewiselin,
                        'piecewise constant' : piecewiseconst,
                        'reopening'          : reopening
                       }

    intervention    = params['interv']
    
    if intervention in interv_functions:
        interv_func = interv_functions[intervention]
        interv = interv_func(x, params)
    else:
        interv = beta0 * np.ones_like(x)
    
    return interv



#x = np.arange(100)
#b0 = piecewiseconst(x, {'segments':0, 'beta0':3/7})
#b1 = piecewiseconst(x, {'segments':1, 't1':20, 'beta0':3.1/7, 'beta1':2.1/7})
#b2 = piecewiseconst(x, {'segments':2, 't1':20, 't2':80, 'beta0':3.2/7, 'beta1':2.2/7, 'beta2':1.3/7})
#plt.plot(x,b0,label='const 0')
#plt.plot(x,b1,label='const 1')
#plt.plot(x,b2,label='const 2')
#plt.grid()
#plt.legend()
#plt.show()


#x = np.arange(100)
#b0 = contact_rate(x, {'interv':'reopening', 'beta0':5/7, 'beta2':0.7/7, 'beta4':2/7, 't1':20, 't2':27, 't3':50, 't4':100})
#plt.plot(x,b0,label='Reopening')
#plt.grid()
#plt.legend()
#plt.show()

#-----------------------------------------------
#time-varying detection rate
def testing_rate(x, params):  

    n  = params['testing_segments'] #number of segments
    ri1 = params['detection_rate']
    ti1 = x[0]

    if n==0:
        r = ri1 * np.ones(len(x))
    else:
        r = np.zeros(len(x))
        i=1
        while i<=n:

            ri = params['detection_rate{}'.format(i)]
            ti = params['testing_time{}'.format(i)]

            r = np.where( (ti1<=x)&(x<=ti), (x-ti1)*(ri-ri1)/(ti-ti1)+ri1, r)
            i=i+1
            ri1=ri
            ti1=ti

        r = np.where(x>=ti,ri,r)
    
    return r

#plt.plot(x, testing_rate(x, {'testing_segments':2, 'detection_rate':5e-2,'testing_time1':20, 'detection_rate1':10e-2, 'testing_time2':30, 'detection_rate2':20e-2}),label='testing')
#plt.grid()
#plt.legend()
#plt.show()


#------------------------------------------
#used to seed the initial infection
def seed_infection(x, params):
    init = params['seed_init']
    halflife = params['seed_halflife']
    return init * np.power(0.5, x/halflife)

#plt.plot(x, seed_infection(x,{'seed_init':10, 'seed_halflife':5}),label='seed')
#plt.grid() 
#plt.legend()
#plt.show()

#--------------------------------
#used by DSEIRF to interpolate delayed values
def interp(t, x, y, col):
    f = interpolate.interp1d(x, y if col==0 else y[:,col], assume_sorted=True)
    return f(t)
    
    
#######################################################
# SIR models
########################################################

#-------------------------------------------------------
# basic daily integration of a modified SIR model
# the function returns a numpy matrix, with a row per day and the following columns (cumulative results since day of inception)
cS   = 0  #Susceptible people
cE   = 1  #Exposed, incubating but not infectious
cI   = 2  #Infectious
cT   = 3  #Testing
cR   = 4  #Recovered after infectious (cumulative)
cC   = 5  #Critical: seriously ill after initial infectious period, will die in the next period; assume isolated, so they are not contaminating other people
cF   = 6  #Fatalities (cumulative)
cP   = 7  #Positive cases (cumulative results of positive tests, recovered people are not included)
cI1  = 8
cI2  = 9   
cNE  = 10  #newly exposed
cNI  = 11  #newly infectious
cNC  = 12  #newly critical
cNT  = 13  #newly tested
cNum = 14  


#-------------------------------------------------------
def SEIRF(x, params):

    population      = params['population']
    e0              = params['e0']
    i0              = params['i0']
    t0              = params['t0']
    p0              = params['p0']
    f0              = params['f0']
    c0              = params['c0']
    beta0           = params['beta0']
    gamma_incub     = params['gamma_incub']
    gamma_infec     = params['gamma_infec']
    gamma_crit      = params['gamma_crit']
    gamma_pos       = params['gamma_pos']
    death_rate      = params['death_rate']
    detection_rate  = params['detection_rate']      
    mixing          = params['mixing']
    intervention    = params['interv']
    
    #array of results, the columns are indexed by cS, cE, etc.
    y = np.zeros((x.size,cNum))

    
    #force introduction of exposed people into the population
    #at a decreasing rate over time
    #the function seed_infection needs 'seed_init' and 'seed_halflife' and calculates init * 0.5^(t/halflife)  (where t=0 is the first day corresonding to min(minD, minP))
    if 'seed' in params and params['seed']==True:
        seed = seed_infection(x, params)
    else:
        seed = np.zeros_like(x)

    interv = contact_rate(x, params)   
    
    detect = testing_rate(x, params)
    
    for i in range(0,x.size):
        
        if i==0:
            
            #initial conditions
            exposed    = e0
            infectious = i0
            testing    = t0
            positives  = p0    
            fatalities = f0    
            critical   = c0
            
            recovered = (f0+c0) / death_rate
            
            susceptible = population - exposed - infectious - critical - fatalities - recovered
          
        else:    
            
            #beta = intervention(i, beta0, beta2, t1, t2)
            beta = interv[i]
            detection_rate = detect[i]
    
            newlyexposed = beta * susceptible * (infectious/population)**mixing  + seed[i]
            newlyinfectious = gamma_incub * exposed
            newlycritical = death_rate * gamma_infec * infectious
            newlyrecovered = (1-death_rate) * gamma_infec * infectious
            newfatalities = gamma_crit * critical
        
            d_susceptible = - newlyexposed
            
            d_exposed = newlyexposed - newlyinfectious

            d_testing = detection_rate * newlyinfectious - gamma_pos * testing
            d_positives = gamma_pos * testing

            d_infectious = newlyinfectious - newlycritical - newlyrecovered
            
            d_recovered = newlyrecovered
            
            d_critical = newlycritical - newfatalities
            
            d_fatalities = newfatalities
            
            susceptible += d_susceptible
            exposed     += d_exposed
            positives   += d_positives
            infectious  += d_infectious
            testing     += d_testing
            recovered   += d_recovered
            critical    += d_critical
            fatalities  += d_fatalities

        y[i,cS] = susceptible
        y[i,cE] = exposed
        y[i,cI] = infectious
        y[i,cT] = testing
        y[i,cR] = recovered
        y[i,cC] = critical
        y[i,cF] = fatalities
        y[i,cP] = positives  
            
    return y            

#-------------------------------------------------------
#initialize the SEIRF model from a given i0 and beta, assuming we are in the early exponential growth with coherent number of exposed, infectious, etc..
# in early stages, when S ~ 1 : assume 
# E = alpha * I and 
# C = b * I 
# I = i0 * exp[(beta0-gamma_infec)*t]
# alpha and b can be solved from the dynamics
    
def init_SEIRF(i0, beta0, constants):

    gamma_infec     = constants['gamma_infec']
    gamma_incub     = constants['gamma_incub']
    gamma_crit      = constants['gamma_crit']
    gamma_pos       = constants['gamma_pos']
    death_rate      = constants['death_rate']
    detection_rate  = constants['detection_rate']
    
    e = gamma_infec/gamma_incub
    disc = (1-e)**2 + 4*e*beta0/gamma_infec
    a = (-(1-e)+math.sqrt(disc))/2
    #print('a:{}'.format(a))
    
    b = death_rate * gamma_infec / (gamma_incub * a + gamma_crit - gamma_infec )
    #print('b:{}'.format(b))
    
    mu = detection_rate * gamma_incub * a / (gamma_incub*a + gamma_pos - gamma_infec)
    
    p = constants.copy()
    p['i0'] = i0
    p['beta0'] = beta0
    p['e0'] = a * i0
    p['c0'] = b * i0
    p['t0'] = mu * i0
    
    return p

def init_SEIRF_doubling(fr, doubling, constants):
    gamma_infec     = constants['gamma_infec']
    gamma_incub     = constants['gamma_incub']
    gamma_crit      = constants['gamma_crit']
    gamma_pos       = constants['gamma_pos']
    death_rate      = constants['death_rate']
    detection_rate  = constants['detection_rate']

    #(1) E' = beta * S * I - gamma_incub * E
    #(2) I' = gamma_incub * E - gamma_infec * I
    #(3) C' = death_rate * gamma_infec * I - gamma_crit * C
    #(4) F' = gamma_crit * C
    #(5) T' = detection_rate * gamma_incub * E - gamma_pos * T
    #(6) P' = gamma_pos * T
    
    # in early stages, when S ~ 1 : assume 
    #(7) E = alpha * I and 
    #(8) C = b * I 
    # from data, calibrate ra, fa to 
    #(9) I' = ra * exp(fa*t)  ; doubling_time = ln(2)/fa
    
    # from (2) and (7):
    #(10) I' = (gamma_incub * alpha - gamma_infec)*I => I = I0 * exp[(gamma_incub * alpha - gamma_infec)*t]
    #(11) by matching growth rate in (9) and (10)  alpha = (fa + gamma_infec) / gamma_incub
    
    #(12) from (1), (7) E' = (beta / alpha - gamma_incub)* E => E = E0 * exp[(beta / alpha - gamma_incub)*t]
    #(13) by matching growth rate in (12) and (10) beta = (fa + gamma_incub)*alpha
    
    fa = math.log(2)/doubling
    
    alpha = (fa + gamma_infec)/gamma_incub
    beta  = (fa + gamma_incub)*alpha
    b = death_rate * gamma_infec / (fa + gamma_crit)
    mu = detection_rate * gamma_incub * alpha / (fa + gamma_pos)

    i0 = fr * (gamma_incub * alpha + gamma_crit - gamma_infec) / (death_rate * gamma_crit * gamma_infec)
    c0 = b * i0
    e0 = alpha * i0
    
    p = constants.copy()
    p['i0']   = i0
    p['beta0'] = beta
    p['e0']   = alpha * i0
    p['c0']   = b * i0
    p['t0']   = mu * i0
    
    return p

#-------------------------------------------------------
#use same columns as SEIRF
def SIRF(x, params):

    population      = params['population']
    i0              = params['i0']
    p0              = params['p0']
    f0              = params['f0']
    gamma           = params['gamma_infec']
    death_rate      = params['death_rate']    
    detection_rate  = params['detection_rate']    
    mixing          = params['mixing']
    
    y = np.zeros((x.size,cNum))

    if 'seed' in params and params['seed']==True:
        seed = seed_infection(x, params)
    else:
        seed = np.zeros_like(x)
    
    #the contact rate beta depends on time
    interv = contact_rate(x, params)   

    #the detection rate of infectious people depends on time
    detect = testing_rate(x, params)
    
#    if intervention=='piecewise linear':
#        interv = piecewiselin(x, params) 
#    else:
#        #intervention=='piecewise constant':
#        interv = piecewiseconst(x, params) 
    
    for i in range(0,x.size):
        
        if i==0:
            
            #initial conditions
            infectious = i0
            positives  = p0    
            fatalities = f0    
            recovered = f0 / death_rate
            
            susceptible = population - infectious  - fatalities - recovered
          
        else:    

            beta = interv[i]
            detection_rate = detect[i]

            newlyinfectious = beta * susceptible * math.pow(infectious,mixing) / population + seed[i]
            newfatalities = death_rate * gamma * infectious
            newlyrecovered = (1-death_rate) * gamma * infectious
        
            d_susceptible = - newlyinfectious
            
            d_positives = detection_rate * newlyinfectious

            d_infectious = newlyinfectious - newlyrecovered - newfatalities
            
            d_recovered = newlyrecovered
            
            d_fatalities = newfatalities
            
            susceptible += d_susceptible
            positives   += d_positives
            infectious  += d_infectious
            recovered   += d_recovered
            fatalities  += d_fatalities

        y[i,cS] = susceptible
        y[i,cI] = infectious
        y[i,cR] = recovered
        y[i,cF] = fatalities
        y[i,cP] = positives  
            
    return y            

#-------------------------------------------------------
def DSEIRF(x, params):

    population      = params['population']
    e0              = params['e0']
    i0              = params['i0']
    t0              = params['t0']
    p0              = params['p0']
    f0              = params['f0']
    c0              = params['c0']
    beta0           = params['beta0']
    gamma_incub     = params['gamma_incub']
    gamma_infec     = params['gamma_infec']
    gamma_crit      = params['gamma_crit']
    gamma_pos       = params['gamma_pos']
    death_rate      = params['death_rate']
    detection_rate  = params['detection_rate']      
    
    #gamma_infec = -(beta0 - gamma_infec)/math.log(gamma_infec)
    #print(gamma_infec)
    
    #number of days in history that we are going to need
    n_incub = math.ceil(1/gamma_incub)
    n_infec = math.ceil(1/gamma_infec)
    n_crit = math.ceil(1/gamma_crit)
    n_pos = math.ceil(1/gamma_pos)
    
    n = max(n_incub, n_infec, n_crit, n_pos)
    hx = np.arange(-n, x.size)  #historical days have negative numbers

    #array of results, the columns are indexed by cS, cE, etc.
    y = np.zeros((x.size + n, cNum))
    
    #print(n, len(x), len(hx), len(y[:,0]))
    
    #the contact rate beta depends on time
    interv = contact_rate(x, params)   
    
    #the detection rate of infectious people depends on time
    detect = testing_rate(x, params)
        
    for i in range(x.size):
        
        if i==0:
            
            #initial conditions
            exposed    = e0
            infectious = i0
            testing    = t0
            positives  = p0    
            fatalities = f0    
            critical   = c0
            
            recovered = (f0+c0) / death_rate
            
            susceptible = population - exposed - infectious - critical - fatalities - recovered
            
            #initialize the history
            newlyexposed     = e0 / (n_incub-1)
            newlyinfectious  = i0 / (n_infec-1)
            newlycritical    = c0 / (n_crit-1)
            newlytested      = t0 / (n_pos-1)
        
            y[n-n_incub:n, cNE] = newlyexposed
            y[n-n_infec:n, cNI] = newlyinfectious
            y[n-n_crit:n, cNC]  = newlycritical
            y[n-n_pos:n, cNT]  = newlytested
            
            #plt.plot(hx, y[:,cNE])
            #plt.plot(hx, y[:,cNI])
            #plt.show()
          
        else:    
            
            #beta = intervention(i, beta0, beta2, t1, t2)
            beta = interv[i]
            detection_rate = detect[i]
    
            newlyexposed = beta * susceptible * infectious / population
        
            newlyinfectious = interp(i-1/gamma_incub, hx, y, cNE)  #people who got infected 1/gamma_incub days ago become infection now y[i+n-n_incub, cNE]
            #print('{} exposed:{} infectious:{}'.format(i, newlyexposed, newlyinfectious))
            
            newlycritical = death_rate * interp(i-1/gamma_infec, hx, y, cNI) #y[i+n-n_infec, cNI]  ##a fraction of infectious people will become critical at the end of the infectious period (and eventually die)
            newlyrecovered = (1-death_rate) * interp(i-1/gamma_infec, hx, y, cNI)  #y[i+n-n_infec, cNI]           ##and the others will recover
            newfatalities = interp(i-1/gamma_crit, hx, y, cNC)  #y[i+n-n_crit, cNC]   # critical people die at the end of the critical stage
        
            d_susceptible = - newlyexposed
            
            d_exposed = newlyexposed - newlyinfectious

            newlytested = detection_rate * newlyinfectious
            newlypositives = interp(i-1/gamma_pos, hx, y, cNT)  #y[i+n-n_pos, cNT]   #people tested 1/gamma_pos days ago get their results now
            d_testing = newlytested - newlypositives
            d_positives = newlypositives

            d_infectious = newlyinfectious - newlycritical - newlyrecovered
            
            d_recovered = newlyrecovered
            
            d_critical = newlycritical - newfatalities
            
            d_fatalities = newfatalities
            
            susceptible += d_susceptible
            exposed     += d_exposed
            positives   += d_positives
            infectious  += d_infectious
            testing     += d_testing
            recovered   += d_recovered
            critical    += d_critical
            fatalities  += d_fatalities

        y[i+n,cS] = susceptible
        y[i+n,cE] = exposed
        y[i+n,cI] = infectious
        y[i+n,cT] = testing
        y[i+n,cR] = recovered
        y[i+n,cC] = critical
        y[i+n,cF] = fatalities
        y[i+n,cP] = positives  
        
        y[i+n,cNE] = newlyexposed  
        y[i+n,cNI] = newlyinfectious  
        y[i+n,cNC] = newlycritical  
        y[i+n,cNT] = newlytested  
            
    
    #plt.plot(hx, y[:,cNE])
    #plt.plot(hx, y[:,cNI])
    #plt.show()
    
    return y[n:,:]    

    
    
    