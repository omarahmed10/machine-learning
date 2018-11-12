# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

## San Francisco Crime Classification

### Introduction

As in all cities, crime is a reality San Francisco: Everyone who lives in San Francisco seems to know someone whose car window has been smashed in, or whose bicycle was stolen within the past year or two. Even Prius’ car batteries are apparently considered fair game by the city’s diligent thieves. The challenge we tackle today involves attempting to guess the class of a crime committed within the city, given the time and location it took place. Such studies are representative of efforts by many police forces today: Using machine learning approaches, one can get an improved understanding of which crimes occur where and when in a city — this then allows for better, dynamic allocation of police resources.

### Domain Background

Supervised learning is one of the most promising field in machine learning and used to achieve many predictions used nowadays even in business goals. Using machine learning techniques to predict the crimes area and category shows the power of technology in the field of safety.
Supervised learning is used in many feild similar to this problem like Predicting Crime Using Time and Location Data in this [paper](http://dspace.bracu.ac.bd/xmlui/bitstream/handle/10361/8197/15141009_CSE.pdf?sequence=1&isAllowed=y) or Weather Forecasting using the weather data of the past two days, which include the maximum temperature, minimum temperature, mean humidity, mean atmospheric pressure, and weather classification for each day in this [paper](http://cs229.stanford.edu/proj2016/report/HolmstromLiuVo-MachineLearningAppliedToWeatherForecasting-report.pdf). 
Also This problem was a challenge on Kaggle on this [link](https://www.kaggle.com/c/sf-crime) and many papers provide solutions for this kind of problem like this [one](https://upcommons.upc.edu/bitstream/handle/2117/96580/MACHINE%20LEARNING%20APPLIED%20TO%20CRIME%20PREDICTION.pdf).

### Problem Statement

This is a Multi-Class Classification problem given the time and location of the crime, We need to predict the category of crime that occurred.

### Solution Statement

Different algorithms will be used to come up with a good result. Each of them will be tried and tested, and finally we will get to see which of them works best for this case.
Cross-validation will be used to validate the models, so the database has to be split into test, train and validation subsets. The resulting train dataset is still too large, and running the testing programs would take too long. To speed up tests and development, we will reduce the database using a clustering algorithm. This algorithm will be K-Means. Then, having the number of elements per cluster, we will be able to decide which element has more weight inside the algorithm. Technically there is no data loss.
Once the data has been treated, the following algorithms will be tried:
- K-Nearest Neighbours
- Neural Networks
