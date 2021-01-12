from datetime import datetime
import boto3
import json

from jinja2 import Environment, FileSystemLoader, select_autoescape

from data import Data
from expgrowth_study import study3
from report import Report

import gc
import traceback

import sys, getopt


import tracemalloc



#------------------------------------------------------------------------
def main(argv):

    debug = False

    try:
        opts, args = getopt.getopt(argv,"hd",["debug"])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt=='-h':
            print('daily_run.py -d')
            sys.exit()
        elif opt in ('-d','--debug'):
            print('DEBUG MODE')
            debug=True
        
    print("BEGIN")

    s3 = boto3.resource('s3')


    d = Data()
    USStates = d.database.jhUS['state'].unique()

    #studies = [ {'region':'US','state':'New York'},
    #            {'region':'US','state':'California'},
    #            {'region':'US','state':'New Jersey'},
    #            {'region':'US','state':'Connecticut'},
    #            {'region':'US','state':'Massachusetts'},
    #            {'region':'US','state':'New Hampshire'},
    #            {'region':'US','state':'Florida'},
    #            {'region':'US','state':'Texas'},
    #            {'region':'US','state':'Georgia'},
    
    studies = [ {'region':'France','state':''},
                {'region':'Italy','state':''},
                {'region':'Spain','state':''},
                {'region':'United Kingdom','state':''},
                {'region':'Portugal','state':''},
                {'region':'Belgium','state':''},
                {'region':'United Kingdom','state':''},
                {'region':'Sweden','state':''},
                {'region':'Netherlands','state':''},
            ]

    studies.append({'region':'US','state':''})
    for state in USStates:
        studies.append({'region':'US','state':state})
        
        
    calcdate = datetime.today().strftime('%Y-%m-%d')

    #---------------------------
    #this is going to be the master index.html, with links to all previous calculations
    ri = Report()  
    ri.set_localfolder('./report/', True)  #set to True to erase previous results on local folder

    #get the list of previous calculations from the S3 bucket
    #create an empty list if the database does not exist
    try:
        s3.Bucket('covid-statistics').download_file('index.json', './report/index.json')
        with open('./report/index.json') as json_file:
            previous_calc = json.load(json_file)
    except:
        previous_calc = {'index':[]}

    #---------------------------
    #the study3() function returns the name of the html page that links to the results
    #this link gets added to the index.html file in the bucket
    links = []
    for s in studies:
        try:

            title = "COVID Statistics for {} {} as of {}".format(s['region'], s['state'], calcdate)

            subfolder = calcdate + '-' + s['region'] + '-' + s['state']


            #study3 should return a Report object that contains the figures to be displayed on the website
            r = study3(source='Johns Hopkins', region=s['region'], state=s['state'], cutoff_positive=1, cutoff_death=1, truncate=0, window=2)


            r.set_localfolder('./report/', erase=False)

            r.to_html(title = title, subfolder=subfolder, index_filename='index.html', template_name='study.html')

            #add a link to the index of calculations for the master index.html
            link = subfolder.replace(" ","_") + '/index.html'
            links.append({'index': link, 'region': s['region'], 'state':s['state']})

            #r.copy_to_s3('covid-statistics')

            del r
            gc.collect()

        except:
            traceback.print_exc()


    #update and save the database of links to calculations results
    previous_calc['index'].insert(0, {'calc_date':calcdate, 'countries':links})
    with open('./report/index.json', 'w') as outfile:
        json.dump(previous_calc, outfile)

    #generate a new index.html page linking to all previous calculation results
    ri.record('previous_calc', previous_calc['index'], '')
    ri.to_html('COVID Statistics', subfolder='', index_filename='index.html', template_name='index.html')

    #copy the whole thing to S3
    if not debug:
        print("COPY TO S3")
        ri.copy_to_s3('covid-statistics')
        

#-------------------------------------
if __name__ == "__main__":
    
    
    tracemalloc.start() #https://docs.python.org/3/library/tracemalloc.html
    snapshot = tracemalloc.take_snapshot()
    
    main(sys.argv[1:])

    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
