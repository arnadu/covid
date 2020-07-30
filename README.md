# COVID: Statistics on COVID-19 
using data curated by Johns Hopkins University (JHU), The Economist, and The COVID Tracking Project

Three types of analyzes are produced by the code in this repo, for a selection of European countries and US States:

- Comparison of COVID fatalities to Excess Deaths. This analysis provides a high-level understanding of the severity of the epidemic in relation to the historical level of daily fatalities.

- Trends of COVID infections and fatalities, assuming separate piece-wise exponential growth regimes for daily positive test counts, and for daily fatalities counts. This analysis provides an understanding of the change of the speed of propagation over time.

- Evolution of the COVID epidemic calibrated with Susceptible-Infected-Recovered (SIR) model. This analysis provides an understanding of the relationship between testing and fatalities counts, and an estimate of the current number of infected and recovered people in the population.

Results:
--------
The results are published at:https://covid-statistics.s3.amazonaws.com/index.html

Code outline:
-------------

The code is installed on a EC2 Anaconda instance, and uploads new results on a daily basis to a S3 bucket configured to serve a static website.

daily_run.py:  scheduled to run every day; runs the studies for the (hardcoded) list of countries, and uploads the results to S3.

expgrowth_study.py: the math to fit a piecewise exponential growth model to a time-series, and the code to apply this math to the daily positive test and daily fatalities curves and produce the plots. The code uses the minimize() function of the scipy.optimize package to implement a maximum likelihood regression using a negative binomial distribution (alpha=0.1).

SIR_study.py: the code to calibrate the SIR model and to produce the plots. The code uses the LMFIT package to do a least-square fit of the SIR model to the data.

SIR.py: the implementation of the SIR model and the time-varying profiles for the contact rate curve that is calibrated to the reported data.

data.py: functions to load data from JHU and other external repositories

report.py and templates: utility code to produce the HTML reports using the JINJA2 templating library



