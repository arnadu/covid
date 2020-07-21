
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from scipy.optimize import minimize
from scipy.special import loggamma
from sklearn.linear_model import LinearRegression

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


#--------------------------------------------------------------
#a function to put a matplotlib figure directly into an AWS S3 bucket
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import boto3
import io

def output_plot(bucket_name, filepath, fig):
    canvas = FigureCanvas(fig) # renders figure onto canvas
    imdata = io.BytesIO() # prepares in-memory binary stream buffer (think of this as a txt file but purely in memory)
    canvas.print_png(imdata) # writes canvas object as a png file to the buffer. You can also use print_jpg, alternatively

    s3 = boto3.resource('s3')#,
                        #aws_access_key_id='your access key id',
                        #aws_secret_access_key='your secret access key',
                        #region_name='us-east-1') # or whatever region your s3 is in
    
    s3.Object(bucket_name, filepath).put(Body=imdata.getvalue(), ContentType='image/png') 
    # this makes a new object in the bucket and puts the file in the bucket
    # ContentType parameter makes sure resulting object is of a 'image/png' type and not a downloadable 'binary/octet-stream'

    #s3.ObjectAcl(bucket_name, filepath).put(ACL='public-read')
    # include this last line if you find the url for the image to be inaccessible



#MAXIMUM LIKELIHOOD
#GENERAL THEORY OF FITTING EPIDEMIOLOGIVAL MODEL: https://www.sciencedirect.com/science/article/pii/S2468042719300491
#INTRO TO MLE, POISSON and NEGATIVE BINOMIAL
#https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
#https://towardsdatascience.com/an-illustrated-guide-to-the-poisson-regression-model-50cccba15958
#https://towardsdatascience.com/negative-binomial-regression-f99031bb25b4
#THEORY OF POISSON+GAMMA MIXTURE and equivalence to Negative Binomial https://gregorygundersen.com/blog/2019/09/16/poisson-gamma-nb/
#Generalized Linear Model and STATSMODELS:
#https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab
#https://www.statsmodels.org/stable/glm.html   
#HOW TO CREATE CUSTOM MODEL FOR STATSMODELS https://austinrochford.com/posts/2015-03-03-mle-python-statsmodels.html 
from scipy import stats
#stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#--------------------------------------------------------------
#various exponential growth models that can be fit using fit_model()
def model_expgrowth_continuousplit(x, params, constants):  
    r0 = params[0]
    b0 = params[1]
    b1 = params[2]
    t1 = constants['t1']
    
    y = np.zeros_like(x)
    
    y = r0 * np.exp(b0 * x)
    
    r1 = y[np.where(x==t1)[0]]
    y = np.where(t1<x, r1 * np.exp(b1 * (x - t1)), y)

    return y

#x = np.arange(10,100)
#y = model_expgrowth_continuousplit(x, [1,2/7,0.8/7],{'t1':40})
#plt.plot(x,y)
#plt.yscale('log')
#plt.show()

def model_expgrowth(x, params, constants):  
    r = params[0]
    b = params[1]
    return np.exp(b * x) * r

def model_expgrowthquad(x, params, constants):  
    r = params[0]
    b2 = params[1]
    b1 = params[2]
    return np.exp(b2 * np.power(x,2) + b1 * x) * r


#--------------------------------------------------------------
#various log-likelihood functions that can be used by fit_model()
def loglik_leastsquarerel(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    leastsquare = np.nansum( (np.log(yhat)-np.log(y))**2  )  
    return leastsquare

def loglik_leastsquare(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    leastsquare = np.nansum( (yhat-y)**2  )  
    return leastsquare

def loglik_poisson(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    negLL = -np.nansum( -yhat + y * np.log(yhat)  )  #removed constant terms for minimization
    return negLL

def loglik_negbin(y, yhat, alpha):
    #y[] are observed values; yhat are model prediction
    r = 1/alpha
    #negLL = -np.nansum( loggamma(y+r) -loggamma(y+1) - loggamma(r) + y*np.log(yhat) + r*np.log(r) - (y+r)*np.log(yhat+r) )
    negLL = -np.nansum( y*np.log(yhat)  - (y+r)*np.log(yhat+r) )  #removed constant terms to speed minimization <more sensitive to initial guess ???
    return negLL       

#------------------------------
#estimate alpha for negative binomial
#https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/
def calc_alpha(data,fit):
    var = (np.power(data-fit,2) - data)/fit
    X = fit[:,np.newaxis]
    ols = LinearRegression(fit_intercept=False).fit(np.nan_to_num(X), np.nan_to_num(var))
    alpha = ols.coef_[0]
    return alpha

#--------------------------------------------------------------
#fit the given model using the given maximum likelihood
def fit_model(x, y, model_func, constants, loglik_func, guess, bounds, alpha=1):
   
    #this function is called by the scipy's minimize()
    #it returns the negative log likelihood of the model prediction given the model parameters (optimization target) and the constants
    #it is closure to access calibration data x (to compute the prediction) and y (to compute the likelihood)
    def regression_func(params, constants):
        #make a prediction using the given model params
        yhat = model_func(x, params, constants)
        # compute negative of likelihood 
        negLL = loglik_func(y, yhat, alpha)
        return negLL    
    
    mle = minimize(regression_func, x0=guess, bounds=bounds, args=constants, method="L-BFGS-B")#, method ="Nelder-Mead")
    #display(mle)
    
    res = model_func(x, mle.x, constants)
    return mle.x, mle.fun, res    #mle.x is the array of calibrated model parameters; mle.fun is the loglikelihood (without constant terms); res is the model forecast for input range

#--------------------------------------------------------------
#find the inflection point in the data by comparing maximum likelihood of all possible inflection points after separately fitting a model to the left and right sides of a candidate inflection point
def findsplit(x, y, model_func, constants, loglik_func, guess, bounds, n_min, n_max, window=0, alpha=1, conf=0.05):
    
    #calculate likelihood of fit at every possible split
    for split in range(n_min, n_max):
        
        params_left, mle_left, res_l = fit_model(x[:split-window], y[:split-window], model_func, constants, loglik_func, guess, bounds, alpha)
        params_right, mle_right, res_r = fit_model(x[split+window:], y[split+window:], model_func, constants, loglik_func, guess, bounds, alpha)

        if (split==n_min) or (mle_left+mle_right <= min_mle):
            min_split = split
            min_mle = mle_left+mle_right
            res_left = res_l
            res_right = res_r
            p_left = params_left
            p_right = params_right
    
    #calculate liklihood of fit over entire range, without split
    params, mle, res = fit_model(x, y, model_func, constants, loglik_func, guess, bounds, alpha)

    #test for significant improvement if we split the range in two
    lr = - 2 * (min_mle - mle)
    p = stats.chi2.sf(lr, 2) #2 more degrees of freedom in split regression than in full regression
    #print('split ll:{:,.0f} full ll:{:,.0f} lr:{:,.0f} p:{}'.format(min_mle, mle, lr, p))
    
    if p>conf:
        min_split = 0
        min_mle = mle
        res_left = res
        res_right = []
        p_left = params
        p_right = []
    
    buff = np.empty(2*window)
    buff[:]=np.nan
    res = np.append(res_left, buff)
    res = np.append(res, res_right)
    
    r = objdict({})
    r.Split = min_split
    r.Stages = []
    r.Stages.append(p_left)
    r.Stages.append(p_right)
    r.Predict = res

    return r #min_split, p_left, p_right, res   #min_split will be zero if the optimal solution is no split; params are in the left variables


#--------------------------------------------------------------
def findsplit_continuous(x, y, loglik_func, guess, bounds, n_min, n_max, alpha=1, conf=0.05):
    
    #calculate likelihood of fit at every possible split
    for split in range(n_min, n_max):
        
        params, mle, res = fit_model(x, y, model_expgrowth_continuousplit, {'t1':split}, loglik_func, guess, bounds, alpha)

        if (split==n_min) or (mle <= min_mle):
            min_split = split
            min_mle = mle
            min_res = res
            min_params = params
 
    r = objdict({})
    r.split = min_split
    r.mle = min_mle
    r.predict = min_res
    r.params = min_params  #r0 b0 b1

    return r






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
    
    
    

#--------------------------------------------------------------
#--------------------------------------------------------------
def study3(filepath, source, region, state, cutoff_positive,cutoff_death, truncate, window):

    print('------------')
    print(source,': ', region, '-', state)
    
    d = Data(source=source, region=region, state=state, county="", cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate)
    ed = d.excess_deaths()
    h = d.hospitalization()
    
    n = len(d.fatalities)
    print('n:{} minD:{}, minP:{}'.format(n,d.minD, d.minP))

    rpt={'state':'{}{}'.format(d.region, d.state)}

    #--------------------------------
    if ed is not None:   
        fig, axs = plt.subplots(1,1,figsize=(12,6))
        axs.plot(d.xd[d.minD+1:], d.dfatalities,'+:', label='daily fatalities, COVID (JHU)')
        axs.plot(ed['date'], ed['total_deaths']/7, "m:", label='daily fatalities, all causes (Economist)')
        axs.plot(ed['date'], ed['excess_deaths']/7, "g:", label='excess deaths compared to historical baseline')
        format_plot(axs,'linear','{}{} reported Covid fatalities compared to excess deaths vs.historical baseline'.format(region,state))
        
        
        fig.savefig('{}_excessdeaths.png'.format(filepath), bbox_inches='tight')
        output_plot('covid-statistics', '{}_excessdeaths.png'.format(filepath), fig)
        
    #--------------------------------
    fig, axs = plt.subplots(3,2,figsize=(18,18))
    
    axs[0][0].set_title('{} {} - daily counts positives'.format(d.region, d.state))
    axs[0][1].set_title('{} {} - daily counts fatalities'.format(d.region, d.state))
    
    axs[0][1].plot(d.xd[d.minD+1:], d.dfatalities,'o:', label='fatalities')
    axs[0][0].plot(d.xd[d.minD+1:], d.dfatalities,'o:', label='fatalities')
    axs[0][0].plot(d.xd[d.minP+1:], d.dpositives,'o:', label='positives')

    
    axs[1][1].plot(d.xd[d.minD+1:], d.dfatalities,'+:', label='fatalities')
    axs[1][0].plot(d.xd[d.minP+1:], d.dpositives,'1:', label='positives')

    if h is not None:
        axs[0][0].plot(h['date'], h['hospitalizedCurrently'],'1:', label='in hospital')
        axs[1][0].plot(h['date'], h['hospitalizedCurrently'],'1:', label='in hospital')

    #--------------------------------
    #find optimal split in Positives data with a continuous piecewise linear exponential growth rate; 
    #the split would happen 14 days after the beginning of the data and cannot be later than 2 weeks than the end of the data
    r = findsplit_continuous(d.x[d.minP+1:], d.dpositives, loglik_negbin, guess=[1,2/7,0.8/7], bounds=[(1e-3,1e6),(-5/7,5/7),(-5/7,5/7)], 
                  n_min=d.minP+14, n_max=len(d.x)-14, 
                  alpha=0.1, conf=0.05)
    
    axs[0][0].plot(d.xd[d.minP+1:], r.predict,'k:')
    axs[1][0].plot(d.xd[d.minP+1:], r.predict,'k:')
    
    residuals_p = d.dpositives - r.predict
    axs[2][0].plot(d.xd[d.minP+1:], residuals_p,'k:')

    rpt['positives t1'] = r.split
    rpt['initial doubling positive'] = math.log(2)/r.params[1]

    #--------------------------------
    recent_window_p = len(d.x) - r.split
    r = findsplit_continuous(d.x[-recent_window_p:], d.dpositives[-recent_window_p:], loglik_negbin, guess=[1,0.8/7,0.8/7], bounds=[(1e-3,1e6),(-5/7,5/7),(-5/7,5/7)], 
                  n_min=r.split+14, n_max=len(d.x)-14, 
                  alpha=0.1, conf=0.05)
    
    axs[0][0].plot(d.xd[-recent_window_p:], r.predict,'r-')
    axs[1][0].plot(d.xd[-recent_window_p:], r.predict,'r-')
    
    residuals_p = d.dpositives[-recent_window_p:] - r.predict
    axs[2][0].plot(d.xd[-recent_window_p:], residuals_p,'r:')
    
    rpt['positives t2'] = r.split
    rpt['intermediate doubing positive'] = math.log(2)/r.params[1]
    rpt['recent doubing positive'] = math.log(2)/r.params[2]
    
    #--------------------------------
    #--------------------------------
    #find optimal split in Positives data with a continuous piecewise linear exponential growth rate; 
    #the split would happen 14 days after the beginning of the data and cannot be later than 2 weeks than the end of the data
    r = findsplit_continuous(d.x[d.minD+1:], d.dfatalities, loglik_negbin, guess=[1,2/7,0.8/7], bounds=[(1e-3,1e6),(-5/7,5/7),(-5/7,5/7)], 
                  n_min=d.minD+14, n_max=len(d.x)-14, 
                  alpha=0.1, conf=0.05)
    
    axs[0][0].plot(d.xd[d.minD+1:], r.predict,'k:')
    axs[0][1].plot(d.xd[d.minD+1:], r.predict,'k:')
    axs[1][1].plot(d.xd[d.minD+1:], r.predict,'k:')
    
    residuals_f = d.dfatalities - r.predict
    axs[2][1].plot(d.xd[d.minD+1:], residuals_f,'k:')    
    
    rpt['fatalities t1'] = r.split
    rpt['initial doubling fatalities'] = math.log(2)/r.params[1]
    
    #--------------------------------
    recent_window_f = len(d.x) - r.split
    r = findsplit_continuous(d.x[-recent_window_f:], d.dfatalities[-recent_window_f:], loglik_negbin, guess=[1,2/7,0.8/7], bounds=[(1e-3,1e6),(-5/7,5/7),(-5/7,5/7)], 
                  n_min=r.split+14, n_max=len(d.x)-14, 
                  alpha=0.1, conf=0.05)
    
    axs[0][0].plot(d.xd[-recent_window_f:], r.predict,'r-')

    axs[0][1].plot(d.xd[-recent_window_f:], r.predict,'r-')
    axs[1][1].plot(d.xd[-recent_window_f:], r.predict,'r-')
    
    residuals_f[-recent_window_f:] = d.dfatalities[-recent_window_f:] - r.predict
    axs[2][1].plot(d.xd[d.minD+1:], residuals_f,'r:')

    rpt['fatalities t2'] = r.split
    rpt['intermediate doubing fatalities'] = math.log(2)/r.params[1]
    rpt['recent doubing fatalities'] = math.log(2)/r.params[2]
    
    #--------------------------------
    
    for idx, (k,v) in enumerate(rpt.items()):
        print('{}\t\t{}'.format(k,v))
              
    #--------------------------------
    
    titles = ['{} {} Daily Counts (log)', '{} {} Daily Fatalities (log)', '{} {} Daily Positives', '{} {} Daily Fatalities', '{} {} Positives Residuals', '{} {} Fatalities Residuals']
    titles = [t.format(d.region, d.state) for t in titles]
    i=0
    for ax in axs:
        for a in ax:
            scale = 'log' if i<2 else 'linear'
            format_plot(a,scale,titles[i])
            i=i+1
    
    fig.savefig('{}_expgrowth.png'.format(filepath), bbox_inches='tight')
    output_plot('covid-statistics', '{}_expgrowth.png'.format(filepath), fig)
    #plt.show()

study3('US-New York', source='Johns Hopkins', region='US', state='New York', cutoff_positive=1, cutoff_death=1, truncate=0, window=2)

