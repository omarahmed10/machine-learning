#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 22:58:49 2018

@author: omar
"""

#Load Data with pandas, and parse the first column into datetime
import pandas as pd 
train = pd.read_csv('all/train.csv', parse_dates = ['Dates'])
test = pd.read_csv('all/test.csv', parse_dates = ['Dates'])

print(train.head())


"""
    Data Analysis
"""
categories = train.groupby('Category')['X'].count()
categories.sort_values(ascending=True).plot(kind='barh') 

# plot categories to day of week
# Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)
train_data = pd.concat([hour, days, district, train], axis=1)
print(train_data.columns)

day_filter = train_data.groupby('DayOfWeek')['X'].count()
day_filter.sort_values(ascending=True).plot(kind='barh')


# plot categories to location

"""
    Data transformation
"""
print (train.head())
print(train.Dates.dt.minute)

# convert Time to Cartesian coordinates
import math
alpha = (2*math.pi)/24
(hours,minutes,seconds) = (train.Dates.dt.hour,train.Dates.dt.minute,train.Dates.dt.second)
time = hours + (minutes / 60) + (seconds / 3600)
TimeX = math.cos(alpha) * time
TimeY = math.sin(alpha) * time
TimeX = pd.DataFrame(data={'TimeX':TimeX})
TimeY = pd.DataFrame(data={'TimeY':TimeY})

# convert Day of week to Cartesian coordinates
from sklearn import preprocessing
daysMapper = preprocessing.LabelEncoder()
dayNumber = daysMapper.fit_transform(train.DayOfWeek)
beta = (2*math.pi)/7
dayX = [ math.cos(beta*d) for d in dayNumber ]
dayY = [ math.sin(beta*d) for d in dayNumber ]
dayX = pd.DataFrame(data={'dayX':dayX})
dayY = pd.DataFrame(data={'dayY':dayY})

# reconstruct the data (triaing and testing) set
category = pd.get_dummies(train.Category)
training_data = train.drop(['Descript', 'Resolution','Category'],axis=1)
feature = pd.concat([dayX,dayY,TimeX,TimeY,training_data.X,training_data.Y],axis=1)
"""
    Dataset split
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, category, test_size=0.2, random_state=42)


"""
    Dataset Clustering
"""


def cluster(data):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    max_score = 0
    range_n_clusters = [5, 4, 3, 2]
    
    for n_cluster in range_n_clusters:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    
        # Predict the cluster for each data point
        preds = kmeans.predict(data)
    
        # Find the cluster centers
        centers = kmeans.cluster_centers_
    
        # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(data, preds)
        
        if score >= max_score:
            best_centers = centers
            max_score = score
    # return the best clustering with the best score.
    return (best_centers,max_score)

all_cat = set(train.Category)
for cat in all_cat:
    crimes_cat = training_data.loc[training_data.Category == cat]
    (centers, score) = cluster(crimes_cat)
    print(cat,"is clusterd to ",centers.shape,"with score",score)
