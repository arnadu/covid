from datetime import datetime

import boto3
import os
import traceback

print(os.environ['CONDA_DEFAULT_ENV'])

calcdate = datetime.today().strftime('%Y-%m-%d %H:%m')

print(calcdate)

s3 = boto3.resource('s3')

try:
    s3.Bucket('covid-statistics').download_file('index.json', './report/test.json')
except:
     traceback.print_exc()
     
