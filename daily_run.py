from datetime import datetime
import boto3

from expgrowth_study import study3


studies = [ {'region':'US','state':'New York'},
            {'region':'US','state':'California'},
            {'region':'US','state':'New Jersey'},
        ]

calcdate = datetime.today().strftime('%Y-%m-%d')

s3 = boto3.resource('s3')
s3.Bucket('covid-statistics').download_file('index.html', 'index.html')

index_file = open("index.html", "a")
index_file.write('<p>{}:'.format(calcdate))
    

#the study3() function returns the name of the html page that links to the results
#this link gets added to the index.html file in the bucket
for s in studies:
    filepath = '{}-{}-{}'.format(calcdate, s['region'], s['state'])
    link = study3(filepath, source='Johns Hopkins', region=s['region'], state=s['state'], cutoff_positive=1, cutoff_death=1, truncate=0, window=2)
    index_file.write('<a href="{}"> | {} {} |</a>'.format(link, s['region'], s['state']))
    
index_file.write('</p>')
index_file.close()
s3.meta.client.upload_file('index.html', 'covid-statistics', 'index.html',ExtraArgs={'ContentType':'text/html'})


