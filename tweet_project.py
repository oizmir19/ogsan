# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:58:44 2020
# twitter project. Final Project, INTL550, spring 2020
@author: toztel17
"""

#search "hak" tweets under Suriyeli hashtag
# define dates

import tweepy
import csv
#import pandas as pd

consumer_key = "tUMmIuqam6p5YKSdSQNEwcWJw"
consumer_secret = "xtfLWLsptgbJ5vTHqr9midEVm1xmmFUKGluXp7S7XIPiH0qSum"
access_token = "862798478-OaxpgffJcTLbXcjXZNCTJ0ue1GMM36vQIUiLAUJV"
access_token_secret = "FkmbvGHOfb7HUTKuF3G3pxhGrWjsEP5de3VXzcIlKw94l"


# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth, wait_on_rate_limit=True)
 
#geocode da girebiliyorsun

# Open/Create a file to append data
#csvFile = open('tweet_data.csv', 'w', newline='')
with open('tweet_data_muhacir3.csv', 'w', newline='') as csvFile:
#Use csv Writer

    csvWriter = csv.writer(csvFile)
    for tweet in tweepy.Cursor(api.search,q="muhacir",count=100,
                           lang="tr",
                           locale= "tr",
                           since="2020-05-05", until = "2020-05-13").items():
        print (tweet.created_at, tweet.text)
        
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
#%% load data

import re
import os
import sys
import numpy as np
import pandas as pd
import codecs                     # load UTF-8 Content
import json                       # load JSON files
import pandas as pd               # Pandas handles dataframes
import numpy as np                # Numpy handles lots of basic maths operations
import matplotlib.pyplot as plt   # Matplotlib for plotting
import seaborn as sns             # Seaborn for beautiful plots
from dateutil import *            # I prefer dateutil for parsing dates
import math                       # transformations
import statsmodels.formula.api as smf  # for doing statistical regression
import statsmodels.api as sm      # access to the wider statsmodels library, including R datasets
from collections import Counter   # Counter is useful for grouping and counting
import scipy
from rdd import rdd
import datetime



hak_all_times = pd.read_csv('hak_all_time.csv')


hak_all_times['date'] = pd.to_datetime(hak_all_times['date'],format='%d-%m-%y')

unique_dates = list(set(hak_all_times['date'])) # all the dates. this is our x variable

unique_dates = pd.Series(unique_dates)
unique_dates.columns= ['date']
unique_dates = unique_dates.sort_values(ascending=True)

unique_dates = unique_dates.reset_index(drop=True) #format indexing

nTweets = [] # number of tweets per date. this is our y variable

for i in range(0,len(unique_dates)):
    nTweets.append(len(hak_all_times.index[hak_all_times['date'] == unique_dates[i]]))
    if len(hak_all_times.index[hak_all_times['date'] == unique_dates[i]]) == 0:
        nTweets.append(0)


## all data

all_data = pd.concat([time1_allData,time2_allData,time3_allData],ignore_index = True)
all_data['date'] = pd.to_datetime(all_data['date'],format='%d-%m-%y')


total_tweets = [] # number of tweets per date. this is our y variable

for i in range(0,len(unique_dates)):
    total_tweets.append(len(all_data.index[all_data['date'] == unique_dates[i]]))



percentages = []

for i in range(0,len(total_tweets)):
    percentages.append(nTweets[i]/total_tweets[i])


    
#ok... now, we set three dates for t1, t2 and t3 to specify cut-offs:
t1 = hak_all_times['date'][33] # any row that is 20-04-28
t2 = hak_all_times['date'][161] # any row that is 20-05-04
# threshold = hak_all_times['date'][33]
# data_rdd = rdd.truncated_data(hak_all_times, 'date', bandwidth1, cut=threshold)


# dummy code the dates#
unique_dates = sorted(unique_dates) # sort dates
dummy_dates = [] # this is our x variable. unique_dates yerine  hak_all_times['date'] ? sonra dataframe'e append etmemiz lazım
for i in range(0,len(unique_dates)):
    if unique_dates[i] < t1:
        dummy_dates.append(-1)
    elif unique_dates[i] >= t1 and unique_dates[i] < t2:
        dummy_dates.append(0)
    elif unique_dates[i] >= t2:
        dummy_dates.append(1)
        
dummy_dates_data = [] # this is our x value

for i in range(0,len(hak_all_times['date'])):
    if hak_all_times['date'][i] < t1:
        dummy_dates_data.append(-1)
    elif hak_all_times['date'][i] >= t1 and hak_all_times['date'][i] < t2:
        dummy_dates_data.append(0)
    elif hak_all_times['date'][i] >= t2:
        dummy_dates_data.append(1)

dummy_dates_data = pd.Series(dummy_dates_data)
#dummy_dates_data = dummy_dates_data.transpose()
dummy_dates_data = dummy_dates_data.to_frame()

hak_all_times =  pd.concat([hak_all_times,dummy_dates_data], axis = 1, ignore_index = True)
#dummy_dates_data  = dummy_dates_data.reshape(len(dummy_dates_data),1)
hak_all_times.columns = ['date','hour','tweet','dummy_date']




unique_dates_number = list(range(-8,8))



udn_series = pd.Series(unique_dates_number)
udn_series = udn_series.transpose()
ntweet_series = pd.Series(nTweets)
ntweet_series = ntweet_series.transpose()
dummy_dates = pd.Series(dummy_dates)
percentages = pd.Series(percentages)



rdd_analyze_data = pd.concat([udn_series,ntweet_series,dummy_dates,percentages],axis=1,ignore_index = True)
rdd_analyze_data.columns = ['unique_dates_num','nTweets','dummy_dates','percentages']

#%% RDD- burayı kullanıyoruz

# dv = percentages

#window = rdd_analyze_data[(rdd_analyze_data['dummy_dates']>-1) & (rdd_analyze_data['dummy_dates']<1)]



#result = smf.ols(formula = "nTweets ~ unique_dates_num + dummy_dates",data = rdd_analyze_data).fit()

result = smf.ols(formula = "percentages ~ unique_dates_num + dummy_dates",data = rdd_analyze_data).fit()


print(result.summary())

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(rdd_analyze_data.unique_dates_num,rdd_analyze_data.nTweets, color="blue")

#l = rdd_analyze_data[rdd_analyze_data.unique_dates_num < 0].unique_dates_num.count()
plt.plot(rdd_analyze_data.unique_dates_num[0:4],result.predict()[0:4], '-', color="r")

plt.plot(rdd_analyze_data.unique_dates_num[4:10],result.predict()[4:10], '-', color="r")
plt.plot(rdd_analyze_data.unique_dates_num[10:15],result.predict()[10:15], '-', color="r")

plt.axvline(x=-4,color="black", linestyle="--")
plt.axvline(x=1,color="black", linestyle="--")
plt.xlabel('days')
plt.ylabel('percentage of tweets')
plt.title("Regression Discontinuity: Number of Tweets by Before, During and After the Incident Week", fontsize="18")


#%% dv = nTweets

result = smf.ols(formula = "nTweets ~ unique_dates_num + dummy_dates",data = rdd_analyze_data).fit()


print(result.summary())

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(rdd_analyze_data.unique_dates_num,rdd_analyze_data.nTweets, color="blue")

#l = rdd_analyze_data[rdd_analyze_data.unique_dates_num < 0].unique_dates_num.count()
plt.plot(rdd_analyze_data.unique_dates_num[0:4],result.predict()[0:4], '-', color="r")

plt.plot(rdd_analyze_data.unique_dates_num[4:10],result.predict()[4:10], '-', color="r")
plt.plot(rdd_analyze_data.unique_dates_num[10:15],result.predict()[10:15], '-', color="r")

plt.axvline(x=-4,color="black", linestyle="--")
plt.axvline(x=1,color="black", linestyle="--")
plt.xlabel('days')
plt.ylabel('percentage of tweets')
plt.title("Regression Discontinuity: Number of Tweets by Before, During and After the Incident Week", fontsize="18")


