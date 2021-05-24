# -*- coding: utf-8 -*-
import pandas as pd
import json  
from pandas import json_normalize  
from datetime import *
from copy import *
import numpy as np
import matplotlib.pyplot as plt
#... rest of the imports

def compDates(s, sd, ed):
    months = {'Jan': 1, 'Feb' : 2, 'Mar': 3 , 'Apr': 4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct': 10 , 'Nov': 11, 'Dec': 12}
    d1, m1, y1 = [str(x) for x in s.split('-')]    
    d1 = int(d1)
    m1 = months[m1]
    y1 = 2000+int(y1)

    b1 = date(y1, m1, d1)
    b2 = date(int(sd[0:4]), int(sd[5:7]), int(sd[8:10]))
    b3 = date(int(ed[0:4]), int(ed[5:7]), int(ed[8:10]))

    if(b2 <= b1 and b1 <= b3):
        return True
    return False


def covidDf(json_file_path):
    d = pd.read_json(json_file_path) 
    df = json_normalize(d['states_daily']) 
    return df


def Q1_1(json_file_path, start_date, end_date):
    
    confirmed_count = 0
    deceased_count = 0
    recovered_count = 0
    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                confirmed_count+= int(df.iloc[i][36])
            elif(df['status'][i] == 'Recovered'):
                recovered_count+= int(df.iloc[i][36])
            else:
                deceased_count+= int(df.iloc[i][36])

    print('confirmed_count: ',confirmed_count, 'recovered_count: ',recovered_count, 'deceased_count: ',deceased_count)
    return confirmed_count, recovered_count, deceased_count

def Q1_2(json_file_path, start_date, end_date):
    confirmed_count = 0
    deceased_count = 0
    recovered_count = 0
    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                confirmed_count+=int(df['dl'][i] )
            elif(df['status'][i] == 'Recovered'):
                recovered_count+= int(df['dl'][i] )
            else:
                deceased_count+= int(df['dl'][i] )


    print('confirmed_count: ',confirmed_count, 'recovered_count: ',recovered_count, 'deceased_count: ',deceased_count)
    return confirmed_count, recovered_count, deceased_count



def Q1_3(json_file_path, start_date, end_date):
    confirmed_count = 0
    deceased_count = 0
    recovered_count = 0
    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                confirmed_count+=int(df['dl'][i] )+int(df['mh'][i] )
            elif(df['status'][i] == 'Recovered'):
                recovered_count+= int(df['dl'][i] )+int(df['mh'][i] )
            else:
                deceased_count+= int(df['dl'][i] )+int(df['mh'][i] )


    print('confirmed_count: ',confirmed_count, 'recovered_count: ',recovered_count, 'deceased_count: ',deceased_count)
    return confirmed_count, recovered_count, deceased_count


def Q1_4(json_file_path, start_date, end_date):

    df = covidDf(json_file_path)
    ndf = deepcopy(df.apply(pd.to_numeric, errors='coerce'))
    [r,c]=(ndf.shape)
    
    D_conf={}
    D_recov={}
    D_dec={}
    L=list(df.columns)
    DD={}

    for i in range(len(L)):
        DD[i]=L[i]

    for i in range(c):
        D_conf[DD[i]]=0
        D_recov[DD[i]]=0
        D_dec[DD[i]]=0

    for i in range(r):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_conf[DD[j]]+=ndf.iloc[i,j]
            elif(df['status'][i] == 'Recovered'):
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_recov[DD[j]]+=ndf.iloc[i,j]
            else:
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_dec[DD[j]]+=ndf.iloc[i,j]
    del D_conf['date']
    del D_conf['status']
    del D_conf['tt']
    del D_conf['un']

    del D_recov['date']
    del D_recov['status']
    del D_recov['tt']
    del D_recov['un']
    
    del D_dec['date']
    del D_dec['status']
    del D_dec['tt']
    del D_dec['un']
    
    d=sorted(D_conf.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    max_conf=[d[0][0]]
    for i in range(1,len(d)):
        if d[i][1]==d[i-1][1]:
            max_conf.append(d[i][0])
        else:
            break

    d=sorted(D_recov.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    max_recov=[d[0][0]]
    for i in range(1,len(d)):
        if d[i][1]==d[i-1][1]:
            max_recov.append(d[i][0])
        else:
            break
        
    d=sorted(D_dec.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    max_dec=[d[0][0]]
    for i in range(1,len(d)):
        if d[i][1]==d[i-1][1]:
            max_dec.append(d[i][0])
        else:
            break

    # print()
    print('Confirmed \n')
    print('Highest affected State is: ',*max_conf)
    print('Highest affected State count is: ',D_conf[max_conf[0]])
    print('Recovered \n')
    print('Highest affected State is: ',*max_recov)
    print('Highest affected State count is: ',D_recov[max_recov[0]])
    print('Deceased \n')
    print('Highest affected State is: ',*max_dec)
    print('Highest affected State count is: ',D_dec[max_dec[0]])


def Q1_5(json_file_path, start_date, end_date):

    df = covidDf(json_file_path)
    ndf = deepcopy(df.apply(pd.to_numeric, errors='coerce'))
    [r,c]=(ndf.shape)
    
    D_conf={}
    D_recov={}
    D_dec={}
    L=list(df.columns)
    DD={}

    for i in range(len(L)):
        DD[i]=L[i]

    for i in range(c):
        D_conf[DD[i]]=0
        D_recov[DD[i]]=0
        D_dec[DD[i]]=0

    for i in range(r):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_conf[DD[j]]+=ndf.iloc[i,j]
            elif(df['status'][i] == 'Recovered'):
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_recov[DD[j]]+=ndf.iloc[i,j]
            else:
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_dec[DD[j]]+=ndf.iloc[i,j]
    del D_conf['date']
    del D_conf['status']
    del D_conf['tt']
    del D_conf['un']

    del D_recov['date']
    del D_recov['status']
    del D_recov['tt']
    del D_recov['un']
    
    del D_dec['date']
    del D_dec['status']
    del D_dec['tt']
    del D_dec['un']
    
    d=sorted(D_conf.items(), key = lambda kv:(kv[1], kv[0]))
    min_conf=[d[0][0]]
    for i in range(1,len(d)):
        if d[i][1]==d[i-1][1]:
            min_conf.append(d[i][0])
        else:
            break
    
    d=sorted(D_recov.items(), key = lambda kv:(kv[1], kv[0]))
    min_recov=[d[0][0]]
    for i in range(1,len(d)):
        if d[i][1]==d[i-1][1]:
            min_recov.append(d[i][0])
        else:
            break
        
    d=sorted(D_dec.items(), key = lambda kv:(kv[1], kv[0]))
    min_dec=[d[0][0]]
    for i in range(1,len(d)):
        if d[i][1]==d[i-1][1]:
            min_dec.append(d[i][0])
        else:
            break

    # print()
    print('Confirmed \n')
    print('Lowest affected State is: ',*min_conf)
    print('Lowest affected State count is: ',D_conf[min_conf[0]])
    print('Recovered \n')
    print('Lowest affected State is: ',*min_recov)
    print('Lowest affected State count is: ',D_recov[min_recov[0]])
    print('Deceased \n')
    print('Lowest affected State is: ',*min_dec)
    print('Lowest affected State count is: ',D_dec[min_dec[0]])


def Q1_6(json_file_path, start_date, end_date):
    
    conf_date=""
    recov_date=""
    dec_date=""

    conf_spike=0
    recov_spike=0
    dec_spike=0


    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                temp=int(df['dl'][i])
                if temp>conf_spike:
                    conf_spike=temp
                    conf_date=df.iloc[i,7]
                
            elif(df['status'][i] == 'Recovered'):
                temp=int(df['dl'][i])
                if temp>recov_spike:
                    recov_spike=temp
                    recov_date=df.iloc[i,7]
                
            else:
                temp=int(df['dl'][i])
                if temp>dec_spike:
                    dec_spike=temp
                    dec_date=df.iloc[i,7]
    
    
    # print()
    print('Confirmed \n')
    print('Day: ',conf_date)
    print('Count: ',conf_spike)
    print('Recovered \n')
    print('Day: ',recov_date)
    print('Count: ',recov_spike)
    print('Deceased \n')
    print('Day: ',dec_date)
    print('Count: ',dec_spike)


def Q1_7(json_file_path, start_date, end_date):
    
    df = covidDf(json_file_path)
    ndf = deepcopy(df.apply(pd.to_numeric, errors='coerce'))
    [r,c]=(ndf.shape)
    
    D_conf={}
    D_recov={}
    D_dec={}
    L=list(df.columns)
    DD={}

    for i in range(len(L)):
        DD[i]=L[i]

    for i in range(c):
        D_conf[DD[i]]=0
        D_recov[DD[i]]=0
        D_dec[DD[i]]=0

    for i in range(r):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_conf[DD[j]]+=ndf.iloc[i,j]
            elif(df['status'][i] == 'Recovered'):
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_recov[DD[j]]+=ndf.iloc[i,j]
            else:
                for j in range(c):
                    if not np.isnan(ndf.iloc[i,j]):
                        D_dec[DD[j]]+=ndf.iloc[i,j]
    del D_conf['date']
    del D_conf['status']
    del D_conf['tt']
    del D_conf['un']

    del D_recov['date']
    del D_recov['status']
    del D_recov['tt']
    del D_recov['un']
    
    del D_dec['date']
    del D_dec['status']
    del D_dec['tt']
    del D_dec['un']

    ACTIVE_D={}    
    for j in D_conf:
        ACTIVE_D[j]=D_conf[j]-D_recov[j]-D_dec[j]
    
    print("Active Cases on end date, statewise-")
    for i in ACTIVE_D:
        print(i,"- ",ACTIVE_D[i])



def Q2_1(json_file_path, start_date, end_date):

    date_list=[]
    confirmed_count = []
    deceased_count = []
    recovered_count = []
    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                date_list.append(df['date'][i])
                if len(confirmed_count)==0:
                    confirmed_count.append(int(df.iloc[i][36]))
                else:
                    confirmed_count.append(confirmed_count[-1]+int(df.iloc[i][36]) )

            elif(df['status'][i] == 'Recovered'):
                if len(recovered_count)==0:
                    recovered_count.append(int(df.iloc[i][36]))
                else:
                    recovered_count.append(recovered_count[-1]+int(df.iloc[i][36]))
            else:
                if len(deceased_count)==0:
                    deceased_count.append(int(df.iloc[i][36]))
                else:
                    deceased_count.append(deceased_count[-1]+int(df.iloc[i][36]))


    l = np.array([confirmed_count, recovered_count, deceased_count])
    l = np.transpose(l)
  
    index_values=date_list

    column_values = ["Confirmed", "Recovered", "Deceased"]
    df = pd.DataFrame(data = l, index=index_values ,columns = column_values) 
    df.plot(kind='area', stacked=False, figsize=(18, 8))
    plt.savefig('total_cases.png')
    plt.show()
    

def Q2_2(json_file_path, start_date, end_date):

    date_list=[]
    confirmed_count = []
    deceased_count = []
    recovered_count = []
    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                date_list.append(df['date'][i])

                if len(confirmed_count)==0:
                    confirmed_count.append(int(df['dl'][i]))
                else:
                    confirmed_count.append(confirmed_count[-1]+ int(df['dl'][i]))

            elif(df['status'][i] == 'Recovered'):
                if len(recovered_count)==0:
                    recovered_count.append(int(df['dl'][i]))
                else:
                    recovered_count.append(recovered_count[-1]+ int(df['dl'][i]))

            else:
                if len(deceased_count)==0:
                    deceased_count.append(int(df['dl'][i]))
                else:
                    deceased_count.append(deceased_count[-1]+ int(df['dl'][i]))

    l = np.array([confirmed_count, recovered_count, deceased_count])
    l = np.transpose(l)
    index_values=date_list
    column_values = ["Confirmed", "Recovered", "Deceased"]
    df = pd.DataFrame(data = l, index=index_values ,columns = column_values) 

    df.plot(kind='area', stacked=False, figsize=(18, 8))
    plt.savefig('total_cases_delhi.png')
    plt.show()


def Q2_3(json_file_path, start_date, end_date):
    
    FF=[]
    date_list=[]

    df = covidDf(json_file_path)

    for i in range(0,len(df),3):
        if(compDates(df['date'][i], start_date, end_date) == True):
            date_list.append(df['date'][i])

            if len(FF)==0:
                FF.append(int(df['tt'][i])-int(df['tt'][i+1])-int(df['tt'][i+2])         )
            else:
                FF.append(FF[-1]+int(df['tt'][i])-int(df['tt'][i+1])-int(df['tt'][i+2])         )

    l=np.array([FF])
    l=np.transpose(l)
    # print(len(l))
    # print(len(date_list))
    column_values = ['active cases'] 
    index_values=date_list
    df = pd.DataFrame(data = l, index=index_values ,columns = column_values) 

    df.plot(kind='area', stacked=False, figsize=(18, 8))
    plt.savefig('total_cases_active.png')
    plt.show()

  

def Q3(json_file_path, start_date, end_date):


    confirmed_count = []
    deceased_count = []
    recovered_count = []
    df = covidDf(json_file_path)

    for i in range(len(df)):
        if(compDates(df['date'][i], start_date, end_date) == True):
            if(df['status'][i] == 'Confirmed'):
                confirmed_count.append(int(df['dl'][i]))
            elif(df['status'][i] == 'Recovered'):
                recovered_count.append(int(df['dl'][i]))
            else:
                deceased_count.append(int(df['dl'][i]))

    X = [i for i in range(1,len(confirmed_count)+1)]
    X = np.asarray(X).reshape(-1,1)

    confirmed_count = np.asarray(confirmed_count).reshape(-1,1)
    deceased_count = np.asarray(deceased_count).reshape(-1,1)
    recovered_count = np.asarray(recovered_count).reshape(-1,1)

    conf = normalform(X,  confirmed_count)
    confirmed_intercept = conf[0][0]
    confirmed_slope = conf[1][0]

    rec = normalform(X,  recovered_count)
    recovered_intercept = rec[0][0]
    recovered_slope = rec[1][0]

    dec = normalform(X,  deceased_count)
    deceased_intercept = dec[0][0]
    deceased_slope = dec[1][0]
    # print("Confirmed intercept:", confirmed_intercept, " Confirmed slope:", confirmed_slope)
    # print("Recovered intercept:", recovered_intercept, " Recovered slope:", recovered_slope)
    # print("Deceased intercept:", deceased_intercept, " Deceased slope:", deceased_slope)

    return confirmed_intercept, confirmed_slope, recovered_intercept, recovered_slope, deceased_intercept, deceased_slope

def normalform(X,y):
    X = np.insert(X, 0 , np.ones(X.shape[0]), axis = 1)
    normaltheta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)), np.matmul(np.transpose(X), y))
    return normaltheta

if __name__ == "__main__":
    # execute only if run as a script
    print('Arjun Lakhera 2018133,  Daksh Thapar 2018137') # Please put this first

    start_date = "2020-03-14"
    end_date = "2020-09-05"
    
    Q1_1('covid.json', start_date, end_date)
    Q1_2('covid.json', start_date, end_date)
    Q1_3('covid.json', start_date, end_date)
    Q1_4('covid.json', start_date, end_date)
    Q1_5('covid.json', start_date, end_date)
    Q1_6('covid.json', start_date, end_date)
    Q1_7('covid.json', start_date, end_date)
    Q2_1('covid.json', start_date, end_date)
    Q2_2('covid.json', start_date, end_date)
    Q2_3('covid.json', start_date, end_date)
    Q3('covid.json', start_date, end_date)

    #... Rest of the functions