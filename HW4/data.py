import requests as rq
import pandas as pd

def url_reader(ticker):
    url='https://www.google.com/finance/getprices?i=60&p=20d&f=d,o,h,l,c,v&df=cpct&q='+str(ticker)[2:-2]
    response=rq.get(url)
    re=response.text
    return re

def csv_saver(string, filename):
    f=open(filename,'w')
    f.write(string)
    f.close()

df=pd.read_csv('F:/Files/DataMining/HW4/companylist.csv',
                         usecols=[0],skiprows=[0], header=None)
list=df.values

for i in range(len(list)):
    string=url_reader(list[i])
    if len(string)>1000:
        filename=str(list[i])[2:-2]+'.csv'
        csv_saver(string,filename)