import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import mpld3

import boto3
import io
import os
import shutil
import mimetypes

from jinja2 import Environment, FileSystemLoader, select_autoescape

from collections import OrderedDict

class Report(OrderedDict):
    
    #create/empty the local folder where html components will be saved on the local machine before copy to the S3 bucket
    def __init__(self, local_folder):
        self.local_folder = local_folder
        shutil.rmtree(local_folder)  
        os.makedirs(local_folder)
   
    #add an item to the report; items will be pre-processed according to their datatypes during the generation of the HTML report
    #-TEXT - no pre-processing
    #-MPLD3 - export a matplotlib figure as a json object with the MPLD3 library; these json objects can then be used by the javascript front-end of the MPLD3 library to display plots in a browser
    def record(self, name, data, datatype):
        self[name] = {'name':name, 'datatype':datatype, 'data':data}

    #use the specified Jinja2 template to generate an html report saved as <index_filename> in local_folder/subfolder
    #pre-process the Report's items according to their datatypes to be displayed by the html page
    def to_html(self, subfolder, index_filename, template_name):

        #results are written into this subfolder, that will be copied to S3
        os.makedirs(os.path.join(self.local_folder, subfolder))
        
        #use mpl3d to create a json file for every matplotlib figure
        #these json files will be copied to S3, to be loaded by a web browser and visualized by the MPLD3 library
        for idx, (key,val) in enumerate(self.items()):
            print('val:',val)
            if val['datatype']=='MPLD3':
                json_filename = subfolder + '/' + key + '.json'
                mpld3.save_json(val['data'], os.path.join(self.local_folder, json_filename))  #save the json file to the local folder, it will be copied to S3
                val['json_filename'] = json_filename  #add the json's filename to the dictionary, it will be use dby the jinja2 template (in the javascript to load json from S3)
                pass
            else:
                pass
            
       #use Jinja2 to instantiate a web page that will load these json files and display the plots with the mpl3d javascript front-end
        env = Environment(
            loader=FileSystemLoader('./covid/templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        template = env.get_template(template_name)
        page = template.render(title='New York statistics as of 2020-07-24', items=self)
        
        file = open(os.path.join(self.local_folder,subfolder+'/'+index_filename),'w')
        file.write(page)
        file.close()
    
           
   
    #recursively copy the local folder to the S3 bucket
    #set the 'content_type' according to the type of file
    def copy_to_s3(self, s3_bucket):
       
        self.s3_bucket = s3_bucket
        
        s3 = boto3.resource('s3')
        
        for root,dirs,files in os.walk(self.local_folder):
            #root = root.replace("\\","/")
            directory_name = root.replace(self.local_folder,"")
            for file in files:
                #print(directory_name+'/'+file)
                filename = os.path.join(root,file)
                content_type = mimetypes.guess_type(filename)[0]
                s3.meta.client.upload_file(filename, s3_bucket, directory_name+'/'+file, ExtraArgs={'ContentType': content_type})
                

    #experiment with exports
    '''
    def plot(self, subfolder):
        fig, axs = plt.subplots(1,1,figsize=(6,6))
        x = np.arange(10)
        y = x*x
        axs.plot(x,y)
        
        os.makedirs(os.path.join(self.local_folder, subfolder))
        
        fig.savefig(os.path.join(self.local_folder, subfolder+'/test_plot.png'))
        mpld3.save_html(fig, os.path.join(self.local_folder, subfolder+'/test_plot.html'))
        mpld3.save_json(fig, os.path.join(self.local_folder, subfolder+'/test_plot.json'))
        
        #use Jinja2 to instantiate a web page from a template
        env = Environment(
            loader=FileSystemLoader('./covid/templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        template = env.get_template('study.html')
        page = template.render(country='New York', calc_date='2020-07-24')
        
        file = open(os.path.join(self.local_folder,subfolder+'/test_mpl3d_plot.html'),'w')
        file.write(page)
        file.close()
    '''
        
        
        
       
       
    

def test_Report():
    
    r = Report('./report/')
    
    #r.plot('test')
    
    
    fig1, axs = plt.subplots(1,1,figsize=(6,6))
    x = np.arange(10)
    y = x*x
    axs.plot(x,y)
    
    fig2, axs = plt.subplots(1,1,figsize=(6,6))
    x = - np.arange(10)
    y = x*x
    axs.plot(x,y)
    
    r.record('Hello', 'Hello World!', 'TEXT')
    r.record('NY', fig1, 'MPLD3')
    r.record('CA', fig2, 'MPLD3')
    r.record('Bye', 'Good bye...', 'TEXT')
    
    r.to_html(subfolder='test', index_filename='index.html', template_name='study.html')

    r.copy_to_s3('covid-statistics')
    

#test_Report()


       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
   
   
