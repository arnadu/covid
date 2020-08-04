
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.linear_model import LinearRegression
from lmfit import Minimizer, Parameters, report_fit


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import WeekdayLocator
from matplotlib.dates import MonthLocator
from matplotlib.dates import AutoDateLocator
from matplotlib.pyplot import cm


from data import Data
from object_dict import objdict
from report import Report

from SIR import *

#---------------------------------------------------------------
#LMFIT Parameters structure does not support string values; these get passed in a separate dict
#this function brings all of the SEIRF params into a single dict
#---------------------------------------------------------------
def merge_params(params, constants):
    p = params.valuesdict()  #params should be a LMFIT Parameters instance; constants should be a dict
    for i, (k,v) in enumerate(constants.items()):
        if k not in p:  #do not override values that may already be in the Parameters array
            p[k]=v
    return p



def lookup_modelfunc(params):
    if 'model' in params:
        models = {'SEIRF':SEIRF, 'SIRF':SIRF, 'DSEIRF':DSEIRF}
        return models[params['model']]
    else:
        return SEIRF

#---------------------------------------------------------------
#function called by LMFIT Minimizer
#the ret parameter is used to select a column from the SEIRF result array; all columns are returned if ret==0
#if data=None, the function returns the values from SEIRF, otherwise it compares the results of SEIRF to the given data and returns the residuals (this is used by LMFIT)
#params should be a LMFIT Parameters instance; constance a dictionary
#---------------------------------------------------------------
def lmfit_SEIRF_both(params, x, constants, data=None):
    
    parvals = merge_params(params, constants)
        
    i0 = parvals['i0']
    beta0 = parvals['beta0']
    model = init_SEIRF(i0, beta0, parvals)
    
    modelfunc = lookup_modelfunc(parvals)
    y = modelfunc(x, model)   #DSEIRF(x, model)
    
    scaleP = parvals['scaleP']
    
    if data is None:
        return y
    else:
        res = np.append(y[:,cF], scaleP * y[:,cP])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
        res = res - data
        
        #check for NaN and print debug info
        if ~np.isfinite(res).all():
            print('#######error')
            print(parvals)
            print(y[:,cF])

        #res(isinf(res)|isnan(res)) = 100e6
        
        return res
    
def lmfit_SEIRF_flex(params, x, constants, data=None):
    
    parvals = merge_params(params, constants)
        
    i0 = parvals['i0']
    beta0 = parvals['beta0']
    model = init_SEIRF(i0, beta0, parvals)
    
    modelfunc = lookup_modelfunc(parvals)
    y = modelfunc(x, model)   #DSEIRF(x, model)
    
    scaleP = parvals['scaleP']
    
    if data is None:
        return y
    else:
           
        if parvals['fatalities_calib'] == 'cumulative':
            res = y[:,cF]
        elif parvals['fatalities_calib'] == 'differences':
            res = np.diff(y[:,cF])
            res = res[-parvals['fatalities_n']:] #make sure we have the same number of days as in the calib data
            n1 = len(res)
        #if fatalities_calib=='' : fatalities are not used in first stage

        if parvals['positives_calib'] == 'cumulative':
            if parvals['fatalities_calib']=='':
                res = y[:,cP]
            else:
                res = np.append(res, scaleP * y[:,cP])
        elif parvals['positives_calib'] == 'differences':
            temp = np.diff(y[:,cP])
            temp = temp[-parvals['positives_n']:]
            if parvals['fatalities_calib']=='':
                res = temp
            else:
                res = np.append(res, scaleP * temp)          
        
        #res = np.append(y[:,cF], scaleP * y[:,cP])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
        #check for NaN and print debug info
        

            
        res = res - data
        

        
        #check for NaN and print debug info
        if ~np.isfinite(res).all():
            print('#######error')
            print(parvals)
            print(res)

        #res(isinf(res)|isnan(res)) = 100e6
        
        return res
        
#---------------------------------------------------------------
#function called by LMFIT Minimizer
#the ret parameter is used to select a column from the SEIRF result array; all columns are returned if ret==0
#if data=None, the function returns the values from SEIRF, otherwise it compares the results of SEIRF to the given data and returns the residuals (this is used by LMFIT)
#params should be a LMFIT Parameters instance; constants a dictionary
#---------------------------------------------------------------
def lmfit_SEIRF(params, x, constants, data=None, ret=0):
    
    parvals = merge_params(params, constants)
        
    i0 = parvals['i0']
    beta0 = parvals['beta0']
    model = init_SEIRF(i0, beta0, parvals)

    modelfunc = lookup_modelfunc(parvals)
    y = modelfunc(x, model)   #DSEIRF(x, model)
    #y = SEIRF(x, model) 
    
    if ret==0:
        return y
    else:
        if data is None:
            return y[:,ret] #ret should be cF, cP, etc... to get
        else:

            #check for NaN and print debug info
            res = y[:,ret]
            if ~np.isfinite(res).all():
                print('#######error')
                print(parvals)
                print(res)
            
            return y[:,ret]-data


#---------------------------------------------------------------
#
#---------------------------------------------------------------
def calibrate_positives(d, constants, startx):
    
    params = Parameters()
    params.add_many( 
#        ('testing_segments', 0,    False),
#        ('gamma_pos',        1/14, False),    
        ('detection_rate',   3e-2, True, 1e-2, 30e-2),
    )
    
    #calibrate using confirmed positives data
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:], constants, d.positives[startx:], cP))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)

    y = lmfit_SEIRF(result.params, d.x[startx:], constants)
    
    p = merge_params(result.params,constants)
    b = contact_rate(d.x[startx:], p)
    
    return p, y, b


#---------------------------------------------------------------
#
#---------------------------------------------------------------
def calibrate_fatalities(d, constants, startx):
    
    params = Parameters()
    params.add_many( 
        ('death_rate',   0.5e-2, True, 0.1e-2, 5e-2),
    )
    
    #calibrate using confirmed positives data
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:], constants, d.fatalities[startx:], cF))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)

    y = lmfit_SEIRF(result.params, d.x[startx:], constants)
    
    p = merge_params(result.params,constants)

    b = contact_rate(d.x[startx:], p)
    
    return p, y, b


#---------------------------------------------------------------
#Piece-wise linear contact rate with a single varying time point
#---------------------------------------------------------------

def html_row(c1, c2, header=False):
    if header:
        return "<tr><th>{}</th><th>{}</th></tr>".format(c1,c2)
    else:
        return "<tr><td>{}</td><td>{}</td></tr>".format(c1,c2)

def report_calib_html(label, d, p, y):
    
    ht = '<table>'
    ht += html_row("Model",label, True)
    ht += html_row("Population","{:,.0f}".format(d.population))
    ht += html_row("P0","{:,.0f}".format(p['p0']))
    ht += html_row("F0","{:,.0f}".format(p['f0']))
    ht += html_row("I0","{:,.0f}".format(p['i0']))

    ht += html_row("Incub","{:,.0f}".format(1/p['gamma_incub']))
    ht += html_row("Infec","{:,.0f}".format(1/p['gamma_infec']))
    ht += html_row("Testing","{:,.0f}".format(1/p['gamma_pos']))
    ht += html_row("Crit","{:,.0f}".format(1/p['gamma_crit']))

    ht += html_row("R0","{:,.1f}".format(p['beta0']/p['gamma_infec']))
    for i in range(1, p['segments']+1):
        if 'beta{}'.format(i) in p:
            ht += html_row("Day {:.0f}".format(p['t{}'.format(i)]), "{:.1f}".format(p['beta{}'.format(i)]/p['gamma_infec']))

    ht += html_row("IFR","{:.1%}".format(p['death_rate']))

    ht += html_row("Detect","{:.0%}".format(p['detection_rate']))
    for i in range(1, p['testing_segments']+1):
            ht += html_row("Day {:.0f}".format(p['testing_time{}'.format(i)]), "{:.1f}".format(p['detection_rate{}'.format(i)]))

    ht += html_row("Currently Incubating","{:,.0f}".format(y[-1,cE]))
    ht += html_row("Currently Infectious","{:,.0f}".format(y[-1,cI]))
    ht += html_row("Currently Testing","{:,.0f}".format(y[-1,cT]))
    ht += html_row("Currently Critical","{:,.0f}".format(y[-1,cC]))
    ht += html_row("Currently Recovered","{:,.0f} / {:.0%}".format(y[-1,cR], y[-1,cR]/d.population))
    ht += '</table><br>'
    
    return ht
    


#---------------------------------------------------------------
#Piece-wise linear contact rate with a single varying time point
#---------------------------------------------------------------
def report_calib(label, d, p, y):
    
    print('=====================')
    print(label, ': ', d.region, ' ', d.state)
    print('=====================')

    print("Population:\t{:,.0f}".format(d.population))
    print("P0:\t\t{:,.0f}".format(p['p0']))
    print("F0:\t\t{:,.0f}".format(p['f0']))
    print("I0:\t\t{:,.0f}".format(p['i0']))

    print("Incub:\t\t{:.1f}".format(1/p['gamma_incub']))
    print("Infec:\t\t{:.1f}".format(1/p['gamma_infec']))
    print("Testing:\t{:.1f}".format(1/p['gamma_pos']))
    print("Crit:\t\t{:.1f}".format(1/p['gamma_crit']))

    print("R0\t\t{:.1f}".format(p['beta0']/p['gamma_infec']))
    for i in range(1, p['segments']+1):
        if 'beta{}'.format(i) in p:
            print("{:.0f}\t\t{:.1f}".format(p['t{}'.format(i)], p['beta{}'.format(i)]/p['gamma_infec']))
        
    print("IFR:\t\t{:.2%}".format(p['death_rate']))
    
    print("Detect:\t\t{:.2%}".format(p['detection_rate']))
    for i in range(1, p['testing_segments']+1):
        print("{:.0f}\t\t{:.1f}".format(p['testing_time{}'.format(i)], p['detection_rate{}'.format(i)]))
    

    print("Currently Incubating:\t{:,.0f}".format(y[-1,cE]))
    print("Currently Infectious:\t{:,.0f}".format(y[-1,cI]))
    print("Currently Testing:\t{:,.0f}".format(y[-1,cT]))
    print("Currently Critical:\t{:,.0f}".format(y[-1,cC]))
    print("Currently Recovered:\t{:,.0f} or {:.0%} of population".format(y[-1,cR], y[-1,cR]/d.population))
    
    
    
#---------------------------------------------------------------
#Piece-wise linear contact rate with a single varying time point
#---------------------------------------------------------------
def calibrate_fatalities_piecewiselinear_one(d, label, border0, border1, startx, overrides={}):
    
    tmin = startx
    tmax = d.x[-1]

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False, 1e-2, 30e-2),
                     ('mixing',         1, False),

                     ('testing_segments', 0, False),
        
                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , True, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , True, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       1         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          2/7       , True, 0.01/7, 15/7),                 
                     ('beta1',          0.7/7     , True, 0.01/7, 15/7),
                     ('t1',             tmin+border0      , True, tmin+border0, tmax-border1),
                   )

    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'piecewise linear',
                  'init_beta':'',
                  'seed': False   
                }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
            
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:],constants, d.fatalities[startx:],cF))
    result = fitter.minimize()
    #result = minimize(residual, params, args=(x, y0, None))

    #result.params.pretty_print()
    #report_fit(result)

    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_positives(d, p, startx)

    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}


#---------------------------------------------------------------
#Smooth step function 
#---------------------------------------------------------------
def calibrate_fatalities_smoothstep(d, label, border0=0, border1=0, window=7, startx=0, overrides={}):
    tmin = startx
    tmax = d.x[-1]
    
    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False, 1e-2, 30e-2),
                     ('mixing',         1, False),

                     ('testing_segments', 0, False),

                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , True, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , True, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       2         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          2/7       , True, 0.01/7, 15/7),                 
                     ('beta2',          0.7/7     , True, 0.01/7, 15/7),
                     ('t1',             tmin      , True, tmin+border0, tmax-border1-window),
                     ('t2',             tmin      , True, tmin+border0+window, tmax-border1, 't1+{}'.format(window)),
                   )

    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'smooth step',
                  'init_beta':'',
                  'seed': False   
                }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
    
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:], constants, d.fatalities[startx:], cF))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)

    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_positives(d, p, startx)

    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}

#---------------------------------------------------------------
#Re-opening function: a smoot-step followed by a linear increase starting a while later
#---------------------------------------------------------------
def calibrate_fatalities_reopening(d, label, border0, border1, window, startx, overrides={}):
    tmin = startx
    tmax = d.x[-1]
    
    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4,                     False),
                     ('gamma_infec',    1/3,                     False),
                     ('gamma_pos',      1/10,                    False, 1/100, 1/2),
                     ('gamma_crit',     1/14,                    False, 1/42, 1/2),
                     ('death_rate',     0.5e-2,                  False, 0.01e-2, 10e-2),

                     ('mixing',         1,                       False),

                     ('testing_segments',       0,               False),
                     ('testing_time1',          50, False), #(tmin+tmax)/2,   False, tmin+21, tmax-21),
                     ('testing_time2',          tmax,            False),
                     ('detection_rate',         3e-2,            True, 1e-2, 20e-2),
                     ('detection_rate1',        3e-2,            True, 1e-2, 20e-2),    
                     ('detection_rate2',        3e-2,            True, 1e-2, 20e-2),   
        
                     ('population',     d.population,            False),
                     ('f0',             d.fatalities[startx],    False, 0, max(10,10*d.fatalities[startx])),
                     ('p0',             d.positives[startx],     False, 0, max(10,10*d.positives[startx])),

                     ('seed_init',      10,                      False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10,                      False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('i0',             10,                      True, 1e-3, 1e6),

                     ('segments',         4,                     False),
                     ('beta0',            3/7,                   True, 0.01/7, 15/7),                 
                     ('beta2',            0.5/7,                 True, 0.01/7, 15/7),                 
                     ('auxbeta4',         0,                   True, 0, 5/7), 
                     ('beta4',            1/7,                   True, 0.01/7, 15/7, 'beta2 + auxbeta4'),                 
        
                     ('t1',               tmin+border0,          True, tmin+border0, tmax-border1 - window),
                     ('t2',               tmin+border0+window,   True, tmin+border0+window, tmax-border1, 't1+{}'.format(window)),
                     ('aux3',             0.0,                  True, 0, 1),
                     ('t3',               tmin+border0+window,   True, tmin+border0+window, tmax-border1, 't2 + (t4-{}-t2)*aux3'.format(border1)),
                     ('t4',               tmax,                  False),

                   )
    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'reopening',
                  'init_beta':'',
                  'seed': False   
                }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
    
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:], constants, d.fatalities[startx:], cF))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)

    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_positives(d, p, startx)

    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}


#---------------------------------------------------
#Piece-wise linear contact rate on regular time grid 
#---------------------------------------------------
def calibrate_fatalities_piecewiselinear_grid(d, label, window, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    windows = int((tmax-tmin) // window)  #number of windows
    extradays = (tmax-tmin) % window  #extra days will be added to the window at the middle of the data

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False),
                     ('mixing',         1, False),


                     ('testing_segments', 0, False),

                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       windows-1         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          2/7       , True, 0.01/7, 15/7),                 
                   )

    ti = startx
    for i in range(1, windows):  #force last window to be constant (built-in property of the 'piecewise linear' function, it remains constant after the last ti)
        ti = ti + window + (extradays if i==windows//2 else 0)
        params.add('t{}'.format(i),value=ti,vary=False)
        params.add('beta{}'.format(i), value= 2/7, vary=True,  min=0.01/7, max=15/7)
        #print(ti)


    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 'interv':'piecewise linear',
                  'init_beta':'',  #first window constant
                  'seed': False   
                }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
            
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:],constants, d.fatalities[startx:],cF))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_positives(d, p, startx)
    
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}
    

#---------------------------------------------------
#Piece-wise linear solves times and beta for a given number of segments 
#---------------------------------------------------
def calibrate_fatalities_piecewiselinear_multiple(d, label, segments, window, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False),
                     ('mixing',         1, False),

                     ('testing_segments', 0, False),

                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       segments         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          2/7       , True, 0.01/7, 15/7),                 
                   )

    params.add('t0',value=tmin, vary=False)
    for i in range(1, segments+1):  

        ti   = tmax - (segments-i) * window
        ti1  = 't{}'.format(i-1)
        auxi = 'aux{}'.format(i-1)
        
        params.add(auxi,value=0.1, vary=True, min=0, max=1)
        params.add('t{}'.format(i), vary=True,expr="{auxi}*({ti}-{ti1}-{window})+{ti1}+{window}".format(auxi=auxi, ti=ti, ti1=ti1, window=window))

        params.add('beta{}'.format(i), value= 2/7, vary=True,  min=0.01/7, max=15/7)
        #print(ti)


    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 'interv':'piecewise linear',
                  'init_beta':'', 
                  'seed': False   
                }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v

    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:],constants, d.fatalities[startx:],cF))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_positives(d, p, startx)
    
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}


#---------------------------------------------------
#Piece-wise linear solves times and beta for a given number of segments 
#---------------------------------------------------
def calibrate_positives_piecewiselinear_multiple(d, label, segments, window, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False),
                     ('mixing',         1, False),

                     ('testing_segments', 0, False),

                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       segments         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          2/7       , True, 0.01/7, 15/7),                 
                   )

    params.add('t0',value=tmin, vary=False)
    for i in range(1, segments+1):  

        ti   = tmax - (segments-i) * window
        ti1  = 't{}'.format(i-1)
        auxi = 'aux{}'.format(i-1)
        
        params.add(auxi,value=0.1, vary=True, min=0, max=1)
        params.add('t{}'.format(i), vary=True,expr="{auxi}*({ti}-{ti1}-{window})+{ti1}+{window}".format(auxi=auxi, ti=ti, ti1=ti1, window=window))

        params.add('beta{}'.format(i), value= 2/7, vary=True,  min=0.01/7, max=15/7)
        #print(ti)


    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 'interv':'piecewise linear',
                  'init_beta':'', 
                  'seed': False   
                }

    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
            
    fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:],constants, d.positives[startx:],cP))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_fatalities(d, p, startx)
    
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}
    
    
#---------------------------------------------------
#Brute force timing for multiple segments 
#---------------------------------------------------
def calibrate_fatalities_piecewiselinear_brute(d, label, segments, border0, border1, window, step, startx):

    tmin = startx
    tmax = d.x[-1]

    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 'interv':'piecewise constant',
                  'init_beta':'', 
                  'seed': False   
                }
    
    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/14, False),
                     ('gamma_crit',     1/14, False),
                     ('death_rate',     0.5e-2, False),
                     ('detection_rate', 5e-2, False),
                     ('mixing',         1, False),

                     ('testing_segments', 0, False),

                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       segments         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          2/7       , True, 0.01/7, 15/7),                 
                   )

    #helper structures to iterate through all possible segments
    #boundaries is a dictionary of the segments t1, t2, ... tn 
    boundaries = {}
    for i in range(1,segments+1):
        ti = 't{}'.format(i)
        boundaries[ti]=border0 + (i-1)*window
        
        params.add(ti, value=boundaries[ti], vary=False)
        params.add('beta{}'.format(i), value= 2/7, vary=True,  min=0.01/7, max=15/7)
        
        
    def reset_val(c,i): #set ti (and all tj for j>i) to their minimum value (given ti-1): tj=tj-1+window for j>=i
        if i<=segments:
            c['t{}'.format(i)] = c['t{}'.format(i-1)] + window
            reset_val(c,i+1)

    def next_val(c, i): #calculate the next value
        c['t{}'.format(i)] = c['t{}'.format(i)] + step
        reset_val(c, i+1)
        if i>1 and c['t{}'.format(segments)] > tmax - border1:
            next_val(c, i-1)

    #iterate through all possible values of t1,...tn and find the optimal contact rates for each configuration; return the minmum solution
    is_first=True
    rpt=[]
    while True:
        #print('--------------')
        #print(constants)
        
        try:

            #set the boundary values
            for idx,(k,v) in enumerate(boundaries.items()):
                params['t{}'.format(i)].set(value=v)
            
            #print(boundaries)
            
            #find the optimal beta for these boundary values
            fitter = Minimizer(lmfit_SEIRF, params, fcn_args=(d.x[startx:],constants, d.fatalities[startx:],cF))
            result = fitter.minimize()
            mle = result.redchi

            if (is_first) or (mle <= min_mle):
                min_params = result.params.copy()
                min_mle = mle
                is_first=False

            rpt.append({'boundaries':boundaries, 'mle':mle})
        except:
            print('######### error', boundaries)
            
        if segments==0:
            break          
        next_val(boundaries, segments)
        if boundaries['t{}'.format(segments)] > tmax - border1:
            break


    fitter = Minimizer(lmfit_SEIRF, min_params, fcn_args=(d.x[startx:], constants, d.fatalities[startx:],cF))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    p = merge_params(result.params, constants)
    
    p, y, b = calibrate_positives(d, p, startx)
    
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}
    
    
    
#---------------------------------------------------------------
#Smooth step function solving on fatalities and positives
#---------------------------------------------------------------
def calibrate_both_smoothstep(d, label, border0, border1, window, startx):
    tmin = startx
    tmax = d.x[-1]
    
    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4,                     False),
                     ('gamma_infec',    1/7,                     False),
                     ('gamma_pos',      1/7,                     True, 1/100, 1/2),
                     ('gamma_crit',     1/21,                    True, 1/42, 1/2),
                     ('death_rate',     0.5e-2,                  False, 0.01e-2, 10e-2),

                     ('mixing',         1,                       False),

                     ('testing_segments',       0,               False),
                     ('testing_time1',          (tmin+tmax)/2,   False, tmin+21, tmax-21),
                     ('testing_time2',          tmax,            False),
                     ('detection_rate',         3e-2,            True, 1e-2, 20e-2),
                     ('detection_rate1',        3e-2,            True, 1e-2, 20e-2),    
                     ('detection_rate2',        3e-2,            True, 1e-2, 20e-2),   
        
                     ('population',     d.population,            False),
                     ('f0',             d.fatalities[startx],    False, 0, max(10,10*d.fatalities[startx])),
                     ('p0',             d.positives[startx],     False, 0, max(10,10*d.positives[startx])),

                     ('seed_init',      10,                      False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10,                      False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       2,                       False),

                     ('i0',             10,                      True, 1e-3, 1e6),
                     ('beta0',          2/7,                     True, 0.01/7, 15/7),                 
                     ('beta2',          0.7/7,                   True, 0.01/7, 15/7),
                     ('t1',             tmin,                    True, tmin+border0, tmax-border1-window),
                     ('t2',             tmin,                    True, tmin+border0+window, tmax-border1, 't1+{}'.format(window)),
                   )

    scaleP = d.fatalities.max()/d.positives.max()
    data = np.append(d.fatalities[startx:], scaleP * d.positives[startx:])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
    #data = np.nan_to_num(data)
    
    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'smooth step',
                  'init_beta':'',
                  'seed': False,
                  'scaleP':scaleP
                }
   
    fitter = Minimizer(lmfit_SEIRF_both, params, fcn_args=(d.x[startx:],constants, data))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    y = lmfit_SEIRF(result.params, d.x[startx:], constants)

    p = merge_params(result.params,constants)
    
    b = contact_rate(d.x[startx:], p)
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}

    
#---------------------------------------------------
#Piece-wise linear solves times and beta for a given number of segments 
#---------------------------------------------------
def calibrate_both_piecewiselinear_multiple(d, label, segments, window, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/7, False, 1/100, 1/2),   #1/14   #6, 1/14
                     ('gamma_crit',     1/35, False, 1/42, 1/2),    #1/14   #6, 1/21
                     ('death_rate',     0.5e-2, False),
                     ('mixing',         1, False),


                     ('testing_segments',       2,               False),
                     ('testing_time1',          50, False), #(tmin+tmax)/2,   False, tmin+21, tmax-21),
                     ('testing_time2',          tmax,            False),
                     ('detection_rate',         5e-2,            True, 1e-2, 20e-2),
                     ('detection_rate1',        5e-2,            True, 1e-2, 20e-2),    
                     ('detection_rate2',        5e-2,            True, 1e-2, 20e-2),   
        
        
                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       segments         , False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          5/7       , True, 0.01/7, 15/7),                 
                   )

    params.add('t0',value=tmin, vary=False)
    for i in range(1, segments+1):  

        ti   = tmax - (segments-i) * window
        ti1  = 't{}'.format(i-1)
        auxi = 'aux{}'.format(i-1)
        
        params.add(auxi,value=0.1, vary=True, min=0, max=1)
        params.add('t{}'.format(i), vary=True,expr="{auxi}*({ti}-{ti1}-{window})+{ti1}+{window}".format(auxi=auxi, ti=ti, ti1=ti1, window=window))

        params.add('beta{}'.format(i), value= 1/7, vary=True,  min=0.01/7, max=15/7)
        #print(ti)

    scaleP = d.fatalities.max()/d.positives.max()
    data = np.append(d.fatalities[startx:], scaleP * d.positives[startx:])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
    #data = np.nan_to_num(data)
    
    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'piecewise linear',
                  'init_beta':'',
                  'seed': False,
                  'scaleP':scaleP
                }
    
    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
   
    fitter = Minimizer(lmfit_SEIRF_both, params, fcn_args=(d.x[startx:],constants, data))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    y = lmfit_SEIRF_both(result.params, d.x[startx:], constants)
    p = merge_params(result.params,constants)    
    b = contact_rate(d.x[startx:], p)
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}

#---------------------------------------------------
#Piece-wise linear solves times and beta for a given number of segments 
#---------------------------------------------------
def calibrate_both_piecewiseconstant_giventimes(d, label, times, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/7, False, 1/100, 1/2),   #1/14   #6, 1/14
                     ('gamma_crit',     1/35, False, 1/42, 1/2),    #1/14   #6, 1/21
                     ('death_rate',     0.5e-2, False),
                     ('mixing',         1, False),


                     ('testing_segments',       0,               False),
                     ('testing_time1',          50, False), #(tmin+tmax)/2,   False, tmin+21, tmax-21),
                     ('testing_time2',          tmax,            False),
                     ('detection_rate',         5e-2,            True, 1e-2, 20e-2),
                     ('detection_rate1',        5e-2,            True, 1e-2, 20e-2),    
                     ('detection_rate2',        5e-2,            True, 1e-2, 20e-2),   
        
        
                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       len(times), False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          5/3       , True, 0.01/3, 15/3),                 
                   )


    for i in range(1, len(times)+1):      
        params.add('t{}'.format(i),value=times[i-1], vary=False)
        params.add('beta{}'.format(i), value= 1/7, vary=True,  min=0.01/3, max=15/3)
        #print(ti)

    scaleP = d.fatalities.max()/d.positives.max()
    data = np.append(d.fatalities[startx:], scaleP * d.positives[startx:])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
    #data = np.nan_to_num(data)
    
    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'piecewise constant',
                  'init_beta':'',
                  'seed': False,
                  'scaleP':scaleP
                }
    
    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
   
    fitter = Minimizer(lmfit_SEIRF_both, params, fcn_args=(d.x[startx:],constants, data))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    y = lmfit_SEIRF_both(result.params, d.x[startx:], constants)
    p = merge_params(result.params,constants)    
    b = contact_rate(d.x[startx:], p)
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}

#---------------------------------------------------
#Piece-wise linear solves times and beta for a given number of segments 
#---------------------------------------------------
def calibrate_piecewiseconstant_giventimes(d, label, fatalities_calib, positives_calib, times, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4, False),
                     ('gamma_infec',    1/7, False),
                     ('gamma_pos',      1/7, False, 1/100, 1/2),   #1/14   #6, 1/14
                     ('gamma_crit',     1/35, False, 1/42, 1/2),    #1/14   #6, 1/21
                     ('death_rate',     0.5e-2, False),
                     ('mixing',         1, False),


                     ('testing_segments',       0,               False),
                     ('testing_time1',          50, False), #(tmin+tmax)/2,   False, tmin+21, tmax-21),
                     ('testing_time2',          tmax,            False),
                     ('detection_rate',         5e-2,            True, 1e-2, 20e-2),
                     ('detection_rate1',        5e-2,            True, 1e-2, 20e-2),    
                     ('detection_rate2',        5e-2,            True, 1e-2, 20e-2),   
        
        
                     ('population',     d.population, False),
                     ('f0',             d.fatalities[startx], False),
                     ('p0',             d.positives[startx], False),

                     ('seed_init',      10        , False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10        , False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('segments',       len(times), False),

                     ('i0',             10        , True, 1e-3, 1e6),
                     ('beta0',          5/3       , True, 0.01/3, 15/3),                 
                   )


    for i in range(1, len(times)+1):      
        params.add('t{}'.format(i),value=times[i-1], vary=False)
        params.add('beta{}'.format(i), value= 1/7, vary=True,  min=0.01/3, max=15/3)
        #print(ti)

        
    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'piecewise constant',
                  'init_beta':'',
                  'seed': False,
                  #'scaleP':scaleP,
                  'fatalities_calib': fatalities_calib,
                  'positives_calib': positives_calib
                }
    
    
    #scaling factor to normalize fatalities and positives so as to not overweigh errors on positives in the calibration
    scaleP = d.fatalities.max()/d.positives.max() 
    constants['scaleP'] = scaleP
    
    if fatalities_calib == 'cumulative':
        data = d.fatalities[startx:]
        constants['fatalities_n'] = len(data) #remember how many data points, to truncate accordingly in lmfit_SEIRF_flex()
    elif fatalities_calib == 'differences':
        data = d.dfatalities
        data = np.nan_to_num(data)
        constants['fatalities_n'] = len(data)
    else:
        constants['fatalities_n'] = 0
    #if fatalities_calib=='' : fatalities are not used in first stage
    
    if positives_calib == 'cumulative':
        if fatalities_calib=='':
            data = d.positives[startx:]
            data = np.nan_to_num(data)
            constants['positives_n'] = len(data)
        else:
            data = np.append(data, scaleP * d.positives[startx:])
            constants['positives_n'] = len(d.positives[startx:])
    elif positives_calib == 'differences':
        if fatalities_calib=='':
            data = d.dpositives
            data = np.nan_to_num(data)
            constants['positives_n'] = len(data)
        else:
            data = np.append(data, scaleP * np.nan_to_num(d.dpositives))
            constants['positives_n'] = len(d.dpositives)
    else:
        constants['positives_n'] = 0
                  
    #scaleP = d.fatalities.max()/d.positives.max()
    #data = np.append(d.fatalities[startx:], scaleP * d.positives[startx:])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
    data = np.nan_to_num(data)
    

    
    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
   
    fitter = Minimizer(lmfit_SEIRF_flex, params, fcn_args=(d.x[startx:],constants, data))
    result = fitter.minimize()
    p = merge_params(result.params, constants)
    
    #result.params.pretty_print()
    #report_fit(result)
    
    #second stage, if either fatalities or positives were not used in first stage
    if fatalities_calib=='':
        p, y, b = calibrate_fatalities(d, p, startx)
    elif positives_calib=='':
        p, y, b = calibrate_positives(d, p, startx)
    else:
        y = lmfit_SEIRF_flex(result.params, d.x[startx:], constants)
        b = contact_rate(d.x[startx:], p)
    
    t = testing_rate(d.x[startx:], p)
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}

#----------------------------------------------------------------
#'reopening' profile: a smoot-step followed by a linear increase starting a while later
#----------------------------------------------------------------
def calibrate_both_reopening(d, label, border0, border1, window, startx, overrides={}):

    tmin = startx
    tmax = d.x[-1]

    #California:
    #gamma_pos 1/12
    #gamma_crit 1/14
    #beta0 2/7
    #beta2 0.5/7
    #beta4 1/7
    
    params = Parameters()
    #(name, value, vary, min, max, expr)
    params.add_many( 
                     ('gamma_incub',    1/4,                     False),
                     ('gamma_infec',    1/7,                     False),
                     ('gamma_pos',      1/14,                    False, 1/100, 1/2),
                     ('gamma_crit',     1/14,                    False, 1/42, 1/2),
                     ('death_rate',     0.5e-2,                  False, 0.01e-2, 10e-2),

                     ('mixing',         1,                       False),

                     ('testing_segments',       0,               False),
                     ('testing_time1',          50, False), #(tmin+tmax)/2,   False, tmin+21, tmax-21),
                     ('testing_time2',          tmax,            False),
                     ('detection_rate',         3e-2,            True, 1e-2, 20e-2),
                     ('detection_rate1',        3e-2,            True, 1e-2, 20e-2),    
                     ('detection_rate2',        3e-2,            True, 1e-2, 20e-2),   
        
                     ('population',     d.population,            False),
                     ('f0',             d.fatalities[startx],    False, 0, max(10,10*d.fatalities[startx])),
                     ('p0',             d.positives[startx],     False, 0, max(10,10*d.positives[startx])),

                     ('seed_init',      10,                      False, 0, 100),#number of exposed cases seeded every day
                     ('seed_halflife',  10,                      False, 1, 100),#days to halve the number of daily seeded exposed cases

                     ('i0',             10,                      True, 1e-3, 1e6),

                     ('segments',         4,                     False),
                     ('beta0',            3/7,                   True, 0.01/7, 15/7),                 
                     ('beta2',            0.5/7,                 True, 0.01/7, 15/7),                 
                     ('auxbeta4',         0,                   True, 0, 5/7), 
                     ('beta4',            1/7,                   True, 0.01/7, 15/7, 'beta2 + auxbeta4'),                 
        
                     ('t1',               tmin+border0,          True, tmin+border0, tmax-border1 - window),
                     ('t2',               tmin+border0+window,   True, tmin+border0+window, tmax-border1, 't1+{}'.format(window)),
                     ('aux3',             0.0,                  True, 0, 1),
                     ('t3',               tmin+border0+window,   True, tmin+border0+window, tmax-border1, 't2 + (t4-{}-t2)*aux3'.format(border1)),
                     ('t4',               tmax,                  False),

                   )

    scaleP = d.fatalities.max()/d.positives.max()
    data = np.append(d.fatalities[startx:], scaleP * d.positives[startx:])  #calibrate on fatalities and positives at the same time; scale positives to not give too much weight to their larger numbers
    
    #lmfit Parameters cannot accept string values so they get passed in a separate argument
    constants = { 
                  'interv' : 'reopening',
                  'init_beta':'',
                  'seed': False,
                  'scaleP':scaleP
                }
    
    for idx, (k,v) in enumerate(overrides.items()):
        if k in params:
            params[k].set(value=v)
        else:
            constants[k]=v
   
    fitter = Minimizer(lmfit_SEIRF_both, params, fcn_args=(d.x[startx:],constants, data))
    result = fitter.minimize()

    #result.params.pretty_print()
    #report_fit(result)
    
    y = lmfit_SEIRF_both(result.params, d.x[startx:], constants)
    p = merge_params(result.params,constants)    
    b = contact_rate(d.x[startx:], p)
    t = testing_rate(d.x[startx:], p)
    
    report_calib(label, d, p, y)
    
    return {'label':label, 'p':p, 'y':y, 'b':b, 't':t, 'startx':startx}
    
    
    
    
    
    



#--------------------------------------------------------------
#--------------------------------------------------------------
def format_plot(ax, scale='linear', title=''):
    
    ax.grid(axis='y', which='both')
    ax.grid(axis='x', which='major')
    ax.legend()
    ax.set_title(title, pad=5)
    
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:.1f}'.format(x)))       
    ax.set_yscale(scale)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,g}'.format(y)))
    if scale=='log':
        ax.set_ylim(bottom=1)

    ax.xaxis.set_major_locator(WeekdayLocator())
    #ax.xaxis.set_minor_locator(WeekdayLocator())
    #ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))    
    ax.tick_params(axis='x', labelrotation=30)  #do not use autofmt_xdate() for subplots
    
    
def report_charts(d, rpt):
    fig,axs = plt.subplots(8,2,figsize=(18,48))

    axs[0][0].plot(d.xd, d.fatalities, '+:', label='Fatalities')
    axs[1][0].plot(d.xd, d.fatalities, '+:', label='Fatalities')
    axs[0][1].plot(d.xd[d.minD+1:], d.dfatalities, '+:', label='Fatalities')
    axs[1][1].plot(d.xd[d.minD+1:], d.dfatalities, '+:', label='Fatalities')

    axs[3][0].plot(d.xd, d.fatalities, '+:', label='Fatalities')
    axs[3][0].plot(d.xd, d.positives, '+:', label='Positives')
    axs[4][0].plot(d.xd, d.positives, '+:', label='Positives')
    axs[3][1].plot(d.xd[d.minD+1:], d.dfatalities, '+:', label='Fatalities')
    axs[3][1].plot(d.xd[d.minP+1:], d.dpositives, '+:', label='Positives')
    axs[4][1].plot(d.xd[d.minP+1:], d.dpositives, '+:', label='Positives')

    colors = plt.cm.jet(np.linspace(0,1,len(rpt)))

    for idx, r in enumerate(rpt):
        label = r['label']
        p = r['p']
        y = r['y']
        b = r['b']
        startx = r['startx']

        axs[0][0].plot(d.xd[startx:], y[:,cF], '-', color = colors[idx], label=label)
        axs[1][0].plot(d.xd[startx:], y[:,cF], '-', color = colors[idx], label=label)
        axs[0][1].plot(d.xd[startx+1:], np.diff(y[:,cF]), '-', color = colors[idx], label=label)
        axs[1][1].plot(d.xd[startx+1:], np.diff(y[:,cF]), '-', color = colors[idx], label=label)

        axs[2][0].plot(d.xd, np.zeros_like(d.x), '-')
        axs[2][0].plot(d.xd[startx:], d.fatalities[startx:] - y[:,cF], ':', color = colors[idx], label=label)
        axs[2][1].plot(d.xd[startx+1:], np.diff(d.fatalities[startx:]) - np.diff(y[:,cF]), ':', color = colors[idx], label=label)

        axs[3][0].plot(d.xd[startx:], y[:,cP], '-', color = colors[idx], label=label)
        axs[4][0].plot(d.xd[startx:], y[:,cP], '-', color = colors[idx], label=label)
        axs[3][1].plot(d.xd[startx+1:], np.diff(y[:,cP]), '-', color = colors[idx], label=label)
        axs[4][1].plot(d.xd[startx+1:], np.diff(y[:,cP]), '-', color = colors[idx], label=label)

        axs[3][0].plot(d.xd[startx:], y[:,cF], '-', color = colors[idx], label=label)
        axs[3][1].plot(d.xd[startx+1:], np.diff(y[:,cF]), '-', color = colors[idx], label=label)


        axs[5][0].plot(d.xd, np.zeros_like(d.x), '-', color = colors[idx])
        axs[5][0].plot(d.xd[startx:], d.positives[startx:] - y[:,cP], '+:', color = colors[idx], label=label)
        axs[5][1].plot(d.xd[startx+1:], np.diff(d.positives[startx:]) - np.diff(y[:,cP]), '+:', color = colors[idx], label=label)

        axs[6][0].plot(d.xd, np.zeros_like(d.x), '-')
        axs[6][1].plot(d.xd, np.zeros_like(d.x), '-')
        axs[6][0].plot(d.xd[startx:], b/p['gamma_infec'], '-', color = colors[idx], label=label)
        axs[6][1].plot(d.xd[startx:], b/p['gamma_infec'], '-', color = colors[idx], label=label)

        if 't' in r:
            axs[7][0].plot(d.xd[startx:], r['t'], '-', color = colors[idx], label=label)
            axs[7][1].plot(d.xd[startx:], r['t'], '-', color = colors[idx], label=label)

    titles = ['{}{} Cumul Fatalities (log)', '{}{} Daily Fatalities (log)', '{}{} Cumul Fatalities (linear)', '{}{} Daily Fatalities (linear)', '{}{} Cumulative Fatalities - Model', '{}{}Daily Fatalities - Model', 
              '{}{} Cumulative (log)', '{}{} Daily (log)', '{}{} Cumulative (linear)', '{}{} Daily (linear)', '{}{} Cumulative Positives - Model', '{}{} Daily Positives - Model', '{}{} R0','{}{} R0','{}{} Detection','{}{} Detection']

    titles = [t.format(d.region, d.state) for t in titles]

    i=0
    for row,ax in enumerate(axs):
        for a in ax:
            scale = 'log' if row==0 or row==3 else 'linear'
            format_plot(a,scale,titles[i])
            i=i+1
    #axs[0][0].set_yscale('log')
    #axs[0][1].set_yscale('log')

    axs[2][0].axhline(linewidth=1)
    axs[2][1].axhline(linewidth=1)

    axs[5][0].axhline(linewidth=1)
    axs[5][1].axhline(linewidth=1)

    return fig

##########################################################
def SIR_study(d, r, window=2, segments=5):
    
    rpt=[]
    
    #-------------------------------
    label = 'piecewise ({})'.format(segments+1)
    calib = calibrate_fatalities_piecewiselinear_multiple(d, label=label, segments=segments, window=window, startx=d.minP, overrides={'model':'SEIRF', 'gamma_incub':1/4, 'gamma_infec':1/3, 'gamma_pos':1/14, 'gamma_crit':1/14})
    rpt.append(calib)
    
    res = report_calib_html(label, d, calib['p'], calib['y'])
    r.record(label, res, 'HTML')    
   
    #-------------------------------
    #label = 'reopening'
    #calib = calibrate_fatalities_reopening(d, label='reopening', border0=10, border1=0, window=2, startx=d.minP, overrides={'model':'SEIRF', 'death_rate':0.5e-2,'gamma_incub':1/4, 'gamma_infec':1/3, 'gamma_pos':1/14, 'gamma_crit':1/14})
    #rpt.append(calib)
    
    #res = report_calib_html(label, d, calib['p'], calib['y'])
    #r.record(label, res, 'HTML')    
    
    #-------------------------------
    fig = report_charts(d, rpt)
    r.record('SIR', fig, 'MPLPNG')    
    
    del rpt

#--------------------------
def test_SIR_study(source, region, state, cutoff_positive,cutoff_death, truncate, window=2, segments=5):

    r = Report()

    d = Data(source=source, region=region, state=state, county="", cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=0)

    SIR_study(d,r,window, segments)




