import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime
from datetime import timedelta  

import urllib, json
import requests
import io

################################################################
################################################################
#To get the population at Country/Region, Province/State or county level; data from Johns Hopkins.
class DataPopulation():

    def __init__(self):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
        r = requests.get(url).content
        self.data = pd.read_csv(io.StringIO(r.decode('utf-8')))  
        self.data['Admin2'] = self.data['Admin2'].fillna('')
        self.data['Province_State'] = self.data['Province_State'].fillna('')
        
        #get a dictionary of state abbreviations states['NY']='New York'
        url = 'https://worldpopulationreview.com/static/states/abbr-name.csv'
        r = requests.get(url).content
        data = pd.read_csv(io.StringIO(r.decode('utf-8')), header=None)  
        self.states = dict(data.values.tolist())
        #display(self.states)
  
    
    def get(self, country_region, province_state, county):
        d = self.data[self.data['Country_Region'] == country_region]
        d= d[d['Province_State'] == province_state]
        d = d[d['Admin2']==county]    
        return d['Population'].sum()

    def report(self, country_region, province_state):
        d  = self.data[self.data['Country_Region']==country_region]
        if province_state != '':
            d = d[d['Province_State']==province_state]
        return d
            
#test
#dp = DataPopulation()
#print(dp.get('US','New York','New York City'))

################################################################
################################################################
def load_nytimes_excessdeaths():
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/excess-deaths/deaths.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))

    data['date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d').copy() 
    return data

def load_economist_excessdeaths():
    url = 'https://raw.githubusercontent.com/TheEconomist/covid-19-excess-deaths-tracker/master/output-data/excess-deaths/all_weekly_excess_deaths.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))

    data['date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d').copy() 
    return data
    
################################################################
################################################################
def load_kaggle_jh():
    train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
    
    train['date'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

    train['death'] = train['Fatalities']
    train['positive'] = train['ConfirmedCases']

    train['Province_State'].fillna('',inplace=True)

    train['state'] = train['Province_State']
#    train.loc[train['Country_Region'].isin(Europe),'state']=train.loc[train['Country_Region'].isin(Europe),'Country_Region']

    train['region'] = train['Country_Region']
#    train.loc[train['Country_Region'].isin(Europe),'region']='EU'

    train['county'] = ''
    
    return train[['region','state','county','date','positive','death']]

################################################################
def load_kaggle_week5():
    
    data = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")
    #display(temp.head())

    data['Province_State'].fillna('',inplace=True)
    data['County'].fillna('',inplace=True)
    data['date'] = data['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

    return data

################################################################
def reformat_week5(data,region, state, county):
    
    c = data[data['Country_Region']==region]
    c = c[c['Province_State']==state]
    c = c[c['County']==county]

    c = c.pivot_table(index =["Country_Region","Province_State","County","date"], columns = "Target", values = "TargetValue", aggfunc = "sum").reset_index()
    c = c.sort_values(by='date',ascending=True)
    #display(c)
    
    c = c.rename(columns={"Fatalities": "death", "ConfirmedCases": "positive", 'Country_Region':'region', 'Province_State':'state', 'County':'county'})
    
    c['death'] = c['death'].cumsum()
    c['positive'] = c['positive'].cumsum()
    
    return c
    
    
################################################################
def load_covidtracking_states(abbrv):
    
    url = 'https://covidtracking.com/api/states/daily'
    
    r = requests.get(url)

    data = pd.DataFrame(r.json())
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    #data = data.fillna(0)

    data['region'] = 'US'
    data['state'] = data['state'].replace(abbrv) #US_States_codes) #replace abbreviation by State's full name
    
    return data

################################################################
def load_jhu_global():
    
    #--------------
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))

    key_columns = ['Province/State','Country/Region','Lat','Long']
    d1 = pd.melt(data, id_vars=key_columns, var_name='date', value_name='death') 
    
    #--------------
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))

    key_columns = ['Province/State','Country/Region','Lat','Long']
    d2 = pd.melt(data, id_vars=key_columns, var_name='date', value_name='positive') 
    
    #--------------
    d3 = d2.merge(d1, how='outer', on=key_columns.append('date'))
    d3['date'] = pd.to_datetime(d3['date'], format='%m/%d/%y').copy()   

    d3['region'] = d3['Country/Region'].fillna('')
    d3['state'] = d3['Province/State'].fillna('')
    d3['county'] = ''
    return d3[['region','state','county','date','positive','death']]    
    
################################################################
def load_jhu_counties():
    
    #--------------
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))

    key_columns = ['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key','Population']
    d1 = pd.melt(data, id_vars=key_columns, var_name='date', value_name='death') 

    #--------------
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))
    
    key_columns = ['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key'] #population is only in the death file
    d2 = pd.melt(data, id_vars=key_columns, var_name='date', value_name='positive') 
    
    #--------------
    d3 = d2.merge(d1, how='outer', on=key_columns.append('date'))
    d3['date'] = pd.to_datetime(d3['date'], format='%m/%d/%y').copy()   
    
    d3['region'] = 'US'
    d3['state'] = d3['Province_State']
    d3['county'] = d3['Admin2'].fillna('')
    return d3[['region','state','county','Population','date','positive','death']]

################################################################
def load_nytimes_counties():
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    r = requests.get(url).content
    data = pd.read_csv(io.StringIO(r.decode('utf-8')))
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').copy()   
    data = data.rename(columns={"deaths": "death", "cases": "positive"})
    data['region'] = 'US'
    return data


    
################################################################
################################################################
class Database():
    
    def __init__(self):
        
        #initialize population database from Johns Hopkins data set
        self.population = DataPopulation()
        
        #load global data from Johns Hopkins University
        self.jhGlobal = load_jhu_global()
        
        #load US county-level data from Johns Hopkins
        self.jhUS = load_jhu_counties()

        self.ctsData = load_covidtracking_states(self.population.states)
        
        #the following two lines work only when running in a Kaggle environment
        #self.kData   = load_kaggle_jh()   #Week4 and before format
        #self.kData5   = load_kaggle_week5() #Week5 format
            
        self.ntData  = load_nytimes_counties()
        
        self.economistExcessDeaths = load_economist_excessdeaths()

        
    def get(self, source, region, state, county, cutoff_positive, cutoff_death, truncate):
        
        population = self.population.get(region, state, county)
        
        if source == 'Johns Hopkins':
            if region == 'US':
                data = self.jhUS
            else:
                data = self.jhGlobal

        if source == 'NY Times':
            data = self.ntData
        
        if source == 'CovidTracking':
            data = self.ctsData
            
        if source == 'Kaggle4':
            data = self.kData
            
        if source == 'Kaggle5':
            data = reformat_week5(self.kData5, region, state, county)
                
        
        c = data[data['region']==region]
        if state != '':
            c = c[c['state']==state]
            if county != '':
                c = c[c['county']==county]

        c = c.groupby(['date']).sum().reset_index()  #aggregate county data to state level
        c = c.sort_values(by='date', ascending=True)

        #find the first date when the positive count cutoff was reached by this STATE, and keep only these days for calibration
        minDateP = c[c['positive']>cutoff_positive]['date'].min()
        minDateD = c[c['death']>cutoff_death]['date'].min()
        minDate = min(minDateP, minDateD)

        #keep only the records after as the earliest cutoff has been reached
        c = c[c['date']>=minDate].copy()  

        #keep only the given number of days from the beginning, or remove the given number of days from the end
        if truncate != 0:
            c = c[:truncate].copy()  #keep only the given number of days

        #calculate the number of days since the cutoff
        c['Days'] = (c['date'] - minDate) / np.timedelta64(1, 'D')

        x = c['Days'].to_numpy().copy()
        positives = c['positive'].to_numpy().copy()
        fatalities = c['death'].to_numpy().copy()

        return population, c['date'], x, positives, fatalities, (minDateP-minDate).days, (minDateD-minDate).days

################################################################
################################################################
class Data():
    
    database = Database()
    
    def __init__(self, source="Johns Hopkins", region="US", state="New York", county="", cutoff_positive=10, cutoff_death=10, truncate=0):
        self.source = source
        self.region = region
        self.state = state
        self.county = county
        self.cutoff_positive = cutoff_positive
        self.cutoff_death = cutoff_death
        self.truncate = truncate

        self.population, self.xd, self.x, self.positives, self.fatalities, self.minP, self.minD = self.database.get(source=source, region=region, state=state, county=county, cutoff_positive=cutoff_positive, cutoff_death=cutoff_death, truncate=truncate)
        self.minDate = self.xd.iat[0]

        self.dfatalities = np.diff(self.fatalities[self.minD:]).astype(float)
        self.dfatalities[self.dfatalities <= 0] = np.nan

        self.dpositives = np.diff(self.positives[self.minP:]).astype(float)
        self.dpositives[self.dpositives <= 0] = np.nan       
        
    def excess_deaths(self):
        data2 = self.database.economistExcessDeaths
        if self.region=='US':
            d2 = data2[(data2['country']=='United States')&(data2['region']==self.state)]
            return d2 if d2.shape[0]>0 else None
        else:
            c = 'Britain' if self.region == 'United Kingdom'else self.region
            d2 = data2[(data2['country']==c)&(data2['region']==c)]   #Italy,Italy
            #d2 = d2.groupby(['date']).sum().reset_index()
            return d2 if d2.shape[0]>0 else None
        
    def hospitalization(self):
        data2 = self.database.ctsData
        if self.region=='US':
            #d2 = data2[(data2['region']=='US')&(data2['state']==self.state)]
            d2 = data2[data2['region']=='US']
            if self.state != '':
                d2 = d2[d2['state']==self.state]

            d2 = d2.groupby(['date']).sum().reset_index()  #aggregate county data to state level
            d2 = d2.sort_values(by='date', ascending=True)
            
            return d2[['date','hospitalizedCurrently']] if d2.shape[0]>0 else None
        else:
            None
        
#test        
#d1 = Data(source='Johns Hopkins', region='US', state='New York', county='', cutoff_positive=1, cutoff_death=1, truncate=0)
#print(d1.hospitalization())
