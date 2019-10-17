#!/usr/bin/env python
# coding: utf-8

# # Business case:
# 
# Use Case: For this data science challenge you are provided with a dataset containing mobility traces of ~500 taxi cabs in San Francisco collected over ~30 days. The format of each mobility trace file is the following - each line contains [latitude, longitude, occupancy, time], e.g.: [37.75134 -122.39488 0 1213084687], where latitude and longitude are in decimal degrees, occupancy shows if a cab has a fare (1 = occupied, 0 = free) and time is in UNIX epoch format.
# The goal of this data science challenge is twofold:
# 1. To calculate the potential for yearly reduction on CO2 emissions, caused by the taxi cabs roaming without passengers. In your calculation please assume that the taxi cab fleet is changing at the rate of 10% per month (from combustion engine-powered vehicles to electric vehicles). Assume also that the average passenger vehicle emits about 404 grams of CO2 per mile.
# 2. To build a predictor for taxi drivers, predicting the next place a passenger will hail a cab.
# 
# Bonus question:
# 3. Identify clusters of taxi cabs that you find being relevant

# # Methodology

# Before applying Data Science techniques, we can make use of various problem solving methodologies and their tools to decompose problems
# 
# ✓ Lean Six Sigma 
# 
# ✓ TRIZ (Theory of Inventive Problem Solving) 
# 
# ✓ CRISP-DM 
# 
# ✓ Design Thinking

# ## CRISP-DM

# ftp://public.dhe.ibm.com/software/analytics/spss/documentation/modeler/18.0/en/ModelerCRISPDM.pdf
# 
# CRISP-DM (CRoss Industry Process for Data Mining) is a process model that describes commonly used approaches that data mining experts use to tackle problems. It is a guide that allows data mining projects to be completed faster with higher quality and less resources. It is an evolutionary and iterative process.

# # STEP 1 – BUSINESS UNDERSTANDING 

# Firstly an in-depth analysis of the business objectives and needs has to be done. Current situation must be accessed and from these insights, the goals of carrying out the processes must be defined. This should follow the setting up of a plan to proceed. In our case, our aim is finding yearly CO2 emission reduction, predicting for taxi drivers, predicting the next place a passenger will hail a cab and Identify clusters of taxi cabs that you find being relevant.

# ### Question 1 - CO2 emission reduction

# In order to find yearly CO2 emission reduction, we need to find distance via longitude and latitude in Miles. It should be in miles metric because in the question it is given by average passenger vehicle emits about 404 grams of CO2 per mile. I use only 10% of .txt files. Within those files I keept only occupancy set to zero. Because in the question asked that by the taxi cabs roaming without passengers. 
# 
# I calculate the distances of all drivers to have the total distance of interest for one month. I multiply the result by 12 and get the yearly distance of interest for the CO2 emissions
# Then, I assume that the taxi cab fleet is changing at the rate of 10% per month. This policy means that the number of taxis with combustion engines is reduced by 10% every month. Accordingly I assume that the distance for CO2 emissions is also reduced by 10% every month. Applying this policy for 12 months gives me the new distance for the CO2 emissions with EVs.
# 
# Then, I take the two distances, divide them by 0.1. Because I only took 10% of the files for my calculation and thus approximate the total distance of all taxis. I get the CO2 reduction by multiplying the total distance with the number of CO2 grams per mile and comparing the difference in terms of percentage.
# 
# 

# ### Question 2 - Next place a passenger will hail a cab (Occupancy Model) - DÜZELTİLECEK

# To predict the next place a passenger will hail a cab, I divided the question in to two. First question is what the next place that cab will go is. Second question is predicting a customer hail the cab.
# 
# In the first question, we can think it as time series problem. I build a model that predict the location in t by using as previous location in t-1, t-2 and t-3. First I create simple neural network model with one input layer, one hidden layer and one output layer for one taxi driver. Results are not good therefore I predict longitude and latitude separately with Gradient Boosting algorithm. In order to increase performance of GBM model, we can change the previous window and hyper-parameter tuning. Because of time constrains I wrote hyper-parameter code but not run it. LSTM could be also very useful for this assignment.
# 
# In the second question, I build a Random Forest Classifier to predict whether next location is a pick-up point or not. I subset 10% of the total data and all data and using timestamp data I generate day of week, hour and time data. I think also adding holiday and weather data will be very useful to our model. So I added holiday data as a flag and retrain Random Forest Classifier model. There is only one holiday in the given period of time which is Memorial Day(26th of May) .Result of both models are same but precision of predicting hailing a cab is increase 1%. Therefore we can use holiday flag for our model and it gives other data such as weather will be valuable.
# 
# Notes: I do not do hyper-parameter optimization because of the time constrain. Other algorithms logistic regression, SVM, XgBoost, light GBM etc. could have been used.

# ### Finding next pick-up point (Final Occupancy Model)

# We can get the next pick-up location by combining the next location predictor for each taxi driver and the occupancy classifier.
# 
# The next location prediction coordinates can be fed into the pick-up point classifier which will indicate the next pick-up location with a certain probability.

# ## Bonus Question - Identify clusters of taxi cabs that you find being relevant. 

# I apply two different method to identify cluster of taxi cabs.
# 
# First is using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) that is a popular unsupervised learning method utilized in model building and machine learning algorithms. I subset 10% of the total data and divided into two in terms of occupancy flag. I looked other cabs where they are driving in the past and where they will go after it. By doing so, I clustered its cabs together based on similarity of behavior.
# 
# My second method is implementing RFM methodology in to this question. I subset 10% of the total data and I found total miles per cab, total occupied miles per cab and average active minutes per day. Using mindset of RFM, I split the data into 4 in terms of quantile. Using this quantiles, I create 7 sample of segments. I gave the definition of segments in the coding phase. 

# # STEP 2 – DATA UNDERSTANDING

# We have mobility traces of ~500 taxi cabs in San Francisco collected over ~30 days. The format of each mobility trace file is the following - each line contains [latitude, longitude, occupancy, time] 

# In[1]:


import random
import datetime
from os import listdir
from os.path import join as jp

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib.pyplot as plt
from math import radians, degrees, sin, cos, asin, acos, sqrt
import tensorflow

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, BatchNormalization, Dropout

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pd.options.mode.chained_assignment = None


# In[2]:


import os 
os.getcwd()


# In[3]:


path = '/Users/mustafaozturk/Desktop/cases/PMI Case 2/cabspottingdata'

all_files = [file_name for file_name in listdir(path) if file_name.endswith('.txt')]
print(f'All files are {len(all_files)}')

PerofFiles = 0.1

random.Random(1923).shuffle(all_files)
SelectedFiles = all_files[:int(PerofFiles * len(all_files))]

print(f'\n{len(SelectedFiles)} randomly selected files')


# # STEP 3 – DATA PREPERATION

# Most data used for data mining was originally collected and preserved for other purposes and needs some refinement before it is ready to use for modeling.
# 
# The data preparation phase includes five tasks. These are
# 
# Selecting data
# 
# Cleaning data
# 
# Constructing data
# 
# Integrating data
# 
# Formatting data

# In[4]:


def ConverttoDF(file_name: str):
    df = pd.read_csv(jp(path, file_name), sep=' ', header=None)
    df.index = file_name.split('.')[0] + "_" + df.index.map(str)
    df.columns = ['Latitude', 'Longitude', 'Occupancy', 'Timestamp']
    df['Taxi'] = file_name.split('.')[0]
    return df

def PreviousCoordinates(df: DataFrame):
    df['PrevLatitude'] = df.shift(1)['Latitude']
    df['PrevLongitude'] = df.shift(1)['Longitude']
    df = df.dropna()
    return df

def EstimatedDistance(row: Series):
    # formulation of finding distance via longitude and latitude in Miles from
    # https://gist.github.com/rochacbruno/2883505#gistcomment-1394026
    Longitude = row['Longitude']
    Latitude = row['Latitude']
    PrevLongitude = row['PrevLongitude']
    PrevLatitude = row['PrevLatitude']
    Longitude, Latitude, PrevLongitude, PrevLatitude = map(radians, 
                                                                    [Longitude, Latitude, 
                                                                    PrevLongitude, PrevLatitude])
    try:
        return 3959 * (acos(sin(Latitude) * sin(PrevLatitude) + cos(Latitude) * cos(PrevLatitude) * 
                                cos(Longitude - PrevLongitude)))
    except:
        return 0.0

def DistanceCalculation(df: DataFrame):
    df['Miles'] = 0.0
    df['Miles'] = df.apply(lambda row: EstimatedDistance(row), axis=1)
    return df

Data = pd.DataFrame()
for f in SelectedFiles:
    df = ConverttoDF(file_name=f)
    df = PreviousCoordinates(df=df)
    df = DistanceCalculation(df=df)
    Data = pd.concat([Data, df])


# In[5]:


Data.head()


# Using describe function, very fast data analysis can be done. 
# 
# When I checked the count field below, I can say that there is no nan value there are no outliers.
# 
# By inspecting the data, I found that some coordinates are places in the Pacific ocean but we assume that this is a  GPS error. 
# 
# Maximum time is 06/10/2008 @ 9:09am (UTC) and minimum time is 05/17/2008 @ 10:00am (UTC)
# 
# Occupancy and vacancy rate seems to be equally distributed. 
# 

# In[6]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
Data.describe()


# Average vacant distance for each taxi driver that I select

# In[7]:


NoPassanger = Data[Data['Occupancy'] == 0]
#Distance Without Passanger
DistanceWOPassanger = NoPassanger.groupby(by=['Taxi'])['Miles'].sum()
print(DistanceWOPassanger)


# # STEP 4 – MODELLING

# After data preparation, our data is already in good shape, and now you can search for useful patterns in your data.
# The modeling phase includes four tasks. These are
# 
# Selecting modeling techniques
# 
# Designing test(s)
# 
# Building model(s)
# 
# Assessing model(s)

# ## Question 1: CO2 Reduction Analysis

# In[8]:


DistanceWOPassangerPerMonth = DistanceWOPassanger.sum()
#We can assume that multiplying monthly distance by 12, we can find the yearly distance.
DistanceWOPassangerPerYear = DistanceWOPassangerPerMonth * 12

print(f'The distance for CO2 in one year (combustion engine-powered vehicles) '
      f'is approx {round(DistanceWOPassangerPerYear)} Miles\n')


#electric vehicles
DistanceWOPassangerPerYearEV = 0.0


#assume that the taxi cab fleet is changing at the rate of 10% per month
for month in range(12):
    if month == 0:
        DistanceWOPassangerPerMonthEV = DistanceWOPassangerPerMonth
        DistanceWOPassangerPerYearEV = DistanceWOPassangerPerMonth
    else:
        DistanceWOPassangerPerMonthEV = DistanceWOPassangerPerMonthEV * 0.9
        DistanceWOPassangerPerYearEV += DistanceWOPassangerPerMonthEV
    print(f'The distance for CO2 after {month} month(s) of'
          f' Electric Vehicles is approx {round(DistanceWOPassangerPerMonthEV, 3)} Miles')
    
print(f'\nThe distance for CO2 after one year of Electric Vehicles '
      f' is approx {round(DistanceWOPassangerPerYearEV)} Miles')


# In[9]:


CO2GramsPerMiles = 404

#CO2 emissions without electronic vehicles

CO2EmissionWOEV = DistanceWOPassangerPerYear / PerofFiles * CO2GramsPerMiles

#CO2 emissions with electronic vehicles

CO2EmissionWEV = DistanceWOPassangerPerYearEV / PerofFiles * CO2GramsPerMiles

CO2Reduction = round((CO2EmissionWOEV - CO2EmissionWEV) / CO2EmissionWOEV, 4)

print(f'The CO2 emissions are reduced by {CO2Reduction * 100} %')


# ## Question 2. To build a predictor for taxi drivers, predicting the next place a passenger will hail a cab.

# In[10]:


#San Francisco coordinate is 37.7749° N, 122.4194° W. 
#Therefore we can use 37 and 122 as offset value in order to normalized our data

lat_offset = 37.0
long_offset = -122.0
taxi_driver = SelectedFiles[0]

datapoints = []
with open(file=jp(path, taxi_driver), mode='r') as f:
    for line in f:
        lat, long, occ, ts = line.split()
        datapoints.append([float(lat) - lat_offset, float(long) - long_offset])


# In[11]:


TaxiDrivers = SelectedFiles[0]

Data = []
with open(file=jp(path, TaxiDrivers), mode='r') as f:
     for line in f:
        Latitude, Longitute, Occupation, TS = line.split()
        Data.append([Latitude, Longitute])


# In order to create a new dataset as a timeseries mindset, I follow below steps

# In[12]:


X = []
Y = []
timeseries = 3
for x in range(len(Data) - timeseries):
    TimeSteps = Data[x:x + timeseries]
    X.append([TS for TimeStep in TimeSteps for TS in TimeStep])
    Y.append(Data[x + timeseries])


# This is a timeseries approach so splitting train and test data should be first n% as train last (1-n) % as test. I select 80-20 split rule.

# In[13]:


TraingTestSize = round(len(X) * 0.8)

X_train = np.array(X[:TraingTestSize])
X_test = np.array(X[TraingTestSize:])

Y_train = np.array(Y[:TraingTestSize])
Y_test = np.array(Y[TraingTestSize:])


# In this step, I create neural network with one input layer, one hidden layer and one output layer. I follow MSE because this is a regression problem. 

# In[14]:


model = Sequential()
model.add(BatchNormalization())
model.add(Dense(4, input_shape=(6,)))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(2))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='Adam')


# In[15]:


history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test), verbose=1)
print(f'MSE on training set {history.history["loss"][-1]}')
print(f'MSE on testing set {history.history["val_loss"][-1]}')


# The neural network is trained for 50 epoches. Bellow, the training and testing losses are plotted. From the graph, we see that the network does not overfit. I exclude MSE of the first 15 epochs. Therefore, the MSE is already small and the difference between the train and test curve is visible. Otherwise, the loss at the first 15 epochs of the training is high and the two curves cannot be readable.

# In[16]:


_, ax = plt.subplots()
ax.plot(history.history['loss'][15:])
ax.plot(history.history['val_loss'][15:])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
ax.set_xticklabels(np.arange(5, 55, 5))
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[17]:


Y_pred = model.predict(X_test)


# ## Question 2: Second Model for predicting next place (GBM)

# In this model I try to predict Latitude and Longitude seperately. After predict the next Latitude and Longitude we can predict next coordinates. I use Gradient Boosting Modelling techniques. 
# 
# In order to understand the Gradient Boosting, first we have to understand what is boosting.
# 
# Boosting: It is an approach where you take random samples of data, however the selection of sample is made more intelligently. We subsequently give more and more weight to hard to classify observations. 
# 
# So Gradient Boosting basically combines weak learners into a single strong learner in an iterative fashion

# ### Calculating Latitude 

# In[18]:


TaxiDrivers = SelectedFiles[0]

DataLat = []
with open(file=jp(path, TaxiDrivers), mode='r') as f:
     for line in f:
        Latitude, Longitute, Occupation, TS = line.split()
        DataLat.append([Latitude])


# In[19]:


XLat = []
YLat = []
timeseries = 3
for x in range(len(DataLat) - timeseries):
    TimeSteps = DataLat[x:x + timeseries]
    XLat.append([TS for TimeStep in TimeSteps for TS in TimeStep])
    YLat.append(DataLat[x + timeseries])


# In[20]:


TraingTestSize = round(len(X) * 0.8)

XLat_train = np.array(XLat[:TraingTestSize])
XLat_test = np.array(XLat[TraingTestSize:])

YLat_train = np.array(YLat[:TraingTestSize])
YLat_test = np.array(YLat[TraingTestSize:])

import numpy as np
XLat_train=XLat_train.astype(np.float)
XLat_test=XLat_test.astype(np.float)
YLat_train=YLat_train.astype(np.float)
YLat_test=YLat_test.astype(np.float)


# In[21]:


from sklearn.ensemble import GradientBoostingRegressor 
gbmregressor=GradientBoostingRegressor(n_estimators=100) 
model_gbm_Lat=gbmregressor.fit(XLat_train, YLat_train) 
pred_gbm_Lat=model_gbm_Lat.predict(XLat_test) 
pred_gbm_Lat


# ### Latitude Model Evaluation

# In[22]:


YLat_test_2=YLat_test.reshape(-1)


# In[23]:


actual_Lat=pd.Series(YLat_test_2)
pred_Lat=pd.Series(pred_gbm_Lat).reset_index()
result_Lat=pd.concat([actual_Lat,pred_Lat],axis=1)


# In[24]:


result_Lat.set_index("index",inplace=True)
result_Lat.index.name=None
result_Lat.columns = ['actual_Lat', 'pred_Lat']
result_Lat


# In[25]:


error_Lat=result_Lat.actual_Lat-result_Lat.pred_Lat

R2_Lat=np.corrcoef(result_Lat.actual_Lat,result_Lat.pred_Lat)
MSE_Lat=(error_Lat**2).mean()
RMSE_Lat=(np.sqrt(MSE_Lat)).mean()
MAE_Lat=(np.abs(error_Lat)).mean()
MAPE_Lat=(np.abs(error_Lat)/result_Lat.actual_Lat).mean()


# In[26]:


R2_Lat,MSE_Lat,RMSE_Lat,MAE_Lat,MAPE_Lat


# ## Calculating Longitude

# In[27]:


TaxiDrivers = SelectedFiles[0]

DataLon = []
with open(file=jp(path, TaxiDrivers), mode='r') as f:
     for line in f:
        Latitude, Longitute, Occupation, TS = line.split()
        DataLon.append([Longitute])


# In[28]:


XLon = []
YLon = []
timeseries = 3
for x in range(len(DataLon) - timeseries):
    TimeSteps = DataLon[x:x + timeseries]
    XLon.append([TS for TimeStep in TimeSteps for TS in TimeStep])
    YLon.append(DataLon[x + timeseries])


# In[29]:


TraingTestSize = round(len(X) * 0.8)

XLon_train = np.array(XLon[:TraingTestSize])
XLon_test = np.array(XLon[TraingTestSize:])

YLon_train = np.array(YLon[:TraingTestSize])
YLon_test = np.array(YLon[TraingTestSize:])

import numpy as np
XLon_train=XLon_train.astype(np.float)
XLon_test=XLon_test.astype(np.float)
YLon_train=YLon_train.astype(np.float)
YLon_test=YLon_test.astype(np.float)


# In[30]:


from sklearn.ensemble import GradientBoostingRegressor 
gbmregressor=GradientBoostingRegressor(n_estimators=100) 
model_gbm_Lon=gbmregressor.fit(XLon_train, YLon_train) 
pred_gbm_Lon=model_gbm_Lon.predict(XLon_test) 
pred_gbm_Lon


# ### Longitude Model Evaluations

# In[31]:


YLon_test_2=YLon_test.reshape(-1)


# In[32]:


actual_Lon=pd.Series(YLon_test_2)
pred_Lon=pd.Series(pred_gbm_Lon).reset_index()
result_Lon=pd.concat([actual_Lon,pred_Lon],axis=1)


# In[33]:


result_Lon.set_index("index",inplace=True)
result_Lon.index.name=None
result_Lon.columns = ['actual_Lon', 'pred_Lon']
result_Lon


# In[34]:


error_Lon=result_Lon.actual_Lon-result_Lon.pred_Lon

R2_Lon=np.corrcoef(result_Lon.actual_Lon,result_Lon.pred_Lon)
MSE_Lon=(error_Lon**2).mean()
RMSE_Lon=(np.sqrt(MSE_Lon)).mean()
MAE_Lon=(np.abs(error_Lon)).mean()
MAPE_Lon=(np.abs(error_Lon)/result_Lon.actual_Lon).mean()


# In[35]:


R2_Lon,MSE_Lon,RMSE_Lon,MAE_Lon,MAPE_Lon


# In[36]:


Y_test_Final = np.concatenate([YLat_test, YLon_test], axis=1)


# In[37]:


Pred_Final = np.concatenate([pred_gbm_Lat.reshape(-1,1), pred_gbm_Lon.reshape(-1,1)], axis=1)


# In[38]:


pred_gbm_Lat


# # STEP 5 – EVALUATION (NN and GBM Models)

# You’ll evaluate not just the models you create but also the process that you used to create them, and their potential for practical use.
# 
# The evaluation phase includes three tasks. These are;
# 
# Evaluating results
# 
# Reviewing the process
# 
# Determining the next steps

# In[39]:


plt.figure(figsize=(20, 20))
plt.subplot(421)
plt.scatter(pd.to_numeric(Y_test[:,0]) + lat_offset, pd.to_numeric(Y_test[:,-1]) + long_offset)
plt.title('NN True trajectory for {}'.format(taxi_driver.split('.')[0]))
plt.ylabel('Latitude')
plt.xlabel('Longitude')
 
plt.subplot(422)
plt.scatter(pd.to_numeric(Y_pred[:,0]) + lat_offset, pd.to_numeric(Y_pred[:,-1]) + long_offset)
plt.title('NN Predicted trajectory {}'.format(taxi_driver.split('.')[0]))
plt.ylabel('Latitude')
plt.xlabel('Longitude')
 

plt.subplot(423)
#plt.scatter(pd.to_numeric(pred_gbm_Lat[:,0]) + lat_offset, pd.to_numeric(pred_gbm_Lat[:,-1]) + long_offset)
plt.scatter(Pred_Final[:,0] + lat_offset, Pred_Final[:,-1] + long_offset)
plt.title('GBM True trajectory {}'.format(taxi_driver.split('.')[0]))
plt.ylabel('Latitude')
plt.xlabel('Longitude')
 
 

plt.subplot(424)
#plt.scatter(pd.to_numeric(pred_gbm_Lat[:,0]) + lat_offset, pd.to_numeric(pred_gbm_Lat[:,-1]) + long_offset)
plt.scatter(Pred_Final[:,0] + lat_offset, Pred_Final[:,-1] + long_offset)
plt.title('GBM Predicted trajectory {}'.format(taxi_driver.split('.')[0]))
plt.ylabel('Latitude')
plt.xlabel('Longitude')

plt.tight_layout()


# In[40]:


TaxiDrivers = SelectedFiles[0]

Time = []
with open(file=jp(path, TaxiDrivers), mode='r') as f:
     for line in f:
        Latitude, Longitute, Occupation, TS = line.split()
        Time.append([TS])


# In[41]:


XTime = []
YTime = []
timeseries = 3
for x in range(len(Time) - timeseries):
    TimeSteps = Time[x:x + timeseries]
    XTime.append([TS for TimeStep in TimeSteps for TS in TimeStep])
    YTime.append(Time[x + timeseries])


# In[42]:


TraingTestSize = round(len(X) * 0.8)

YTime_test = np.array(YTime[TraingTestSize:])

import numpy as np
YTime_test=YTime_test.astype(np.float)


# I also plot in the y-axis as a first dimension is Latitude, as a secondary dimension is Longitude. In the x-axis, I add time. Therefore, we can follow minutes by minutes latitude and longitude change. As a result, prediction looks good!

# In[43]:


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(YTime_test, YLat_test, 'g-')
ax2.plot(YTime_test, YLon_test, 'b-')

ax1.set_xlabel('Time')
ax1.set_ylabel('Latitude', color='g')
ax2.set_ylabel('Longitude', color='b')

fig1, ax3 = plt.subplots()


ax4 = ax3.twinx()
ax3.plot(YTime_test, pred_gbm_Lat, 'g-')
ax4.plot(YTime_test, pred_gbm_Lon, 'b-')

ax3.set_xlabel('Time')
ax3.set_ylabel('Latitude', color='g')
ax4.set_ylabel('Longitude', color='b')


# In[44]:


print(f'MSE on testing set for NN model {history.history["val_loss"][-1]}')
print(f'MSE on testing set for GBM Latitude model {MSE_Lat}')
print(f'MSE on testing set for GBM Longitude model {MSE_Lon}')


# First column is exact route for new_ajsnedsi taxi. Second column is predictions of models Neural Network and Gradient Boosting respectively. As you can see in previous plot GBM results are much better than Neural Network model. Also, when we compare Mean Squared Error of NN and GBM's Longitude, Latitude models GBM models’ results are much better than NN model.
# 
# Note: The mean squared error tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. The squaring is necessary to remove any negative signs. It also gives more weight to larger differences. It’s called the mean squared error as you’re finding the average of a set of errors.
# 

# ### Hyperparameter Tuning (did not perform)

# After selecting best algorithm, I am presenting with design choices as to how to define my model architecture. In order to find an optimal model architecture should be for a given model, I have to explore a range of possibilities. In true machine learning fashion, we'll ideally ask the machine to perform this exploration and select the optimal model architecture automatically. Parameters which define the model architecture are referred to as hyperparameters and thus this process of searching for the ideal model architecture is referred to as hyperparameter tuning. GridSearchCV is one of this approach.
# 
# In GridSearchCV approach, machine learning model is evaluated for a range of hyperparameter values. This approach is called GridSearchCV, because it searches for best set of hyperparameters from a grid of hyperparameters values. This entire step is not perform because of time restiction. But I wrote coding steps of hyperparamet tuning below. 

# #### Longitude hyperparameter

# In[45]:


#from sklearn.model_selection import ShuffleSplit

#from sklearn.model_selection import GridSearchCV


# In[46]:


#XLon_train=XLon_train.astype(np.float)

#XLon_test=XLon_test.astype(np.float)

#YLon_train=YLon_train.astype(np.float)

#YLon_test


# In[47]:


#def GradientBooster(param_grid, n_jobs):
    
 #   estimator = GradientBoostingRegressor()
    
  #  cv = ShuffleSplit(XLon_train.shape[0], test_size=0.2)
    
    #classifier = GridSearchCV(estimator=estimator,cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    
    #classifier.fit(XLon_train, YLon_train)
    
    #return cv, classifier.best_estimator_
    


# In[48]:


#param_grid={'n_estimators':[100], 

#           'learning_rate': [0.1, 0.05, 0.02, 0.01],
            
#          'max_depth':[6],
            
#          'min_samples_leaf':[3,5,9,17],
            
#         'max_features':[1.0,0.3,0.1]
            
#          }

#n_jobs=4

#cv,best_est=GradientBooster(param_grid, n_jobs)


# In[49]:


#print "Best Estimator Parameters"

#print"---------------------------"

#print "n_estimators: %d" %best_est.n_estimators

#print "max_depth: %d" %best_est.max_depth

#print "Learning Rate: %.1f" %best_est.learning_rate

#print "min_samples_leaf: %d" %best_est.min_samples_leaf

#print "max_features: %.1f" %best_est.max_features

#print 

#print "Train R-squared: %.2f" %best_est.score(XLon_train,YLon_train)


# In[50]:


#import numpy as np

#import matplotlib.pyplot as plt

#from sklearn import cross_validation

#from sklearn.naive_bayes import GaussianNB 

#from sklearn.datasets import load_digits 

#from sklearn.learning_curve import learning_curve 

#def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, 

#        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

#    plt.figure() 
    
#    plt.title(title) 
    
#    if ylim is not None: 
    
#        plt.ylim(*ylim) 
        
#        plt.xlabel("Training examples") 
        
#        plt.ylabel("Score") 
        
#        train_sizes, 
        
#        train_scores, 
        
#        test_scores = learning_curve( estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) 
        
#        train_scores_mean = np.mean(train_scores, axis=1) 
        
#        train_scores_std = np.std(train_scores, axis=1) 
        
#        test_scores_mean = np.mean(test_scores, axis=1) 
        
#        test_scores_std = np.std(test_scores, axis=1) 
        
#        plt.grid() 
        
#        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r") 
          
#        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g") 
        
#        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score") 
        
#        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score") 
        
#        plt.legend(loc="best") 
        
#        return plt


# In[51]:


#title = "Learning Curves (Gradient Boosted Regression Trees)" 

#estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth, 
#                                      learning_rate=best_est.learning_rate, 
                                      
#min_samples_leaf=best_est.min_samples_leaf, 
#                                      max_features=best_est.max_features)

#plot_learning_curve(estimator, title, XLon_train, YLon_train, cv=cv, n_jobs=n_jobs)

#plt.show()


# In[52]:


##Switching back to the best model from gridsearch 
#estimator = best_est 
##Re-fitting to the train set 
#estimator.fit(XLon_train, YLon_train) 
##Calculating train/test scores - R-squared value 
#print "Train R-squared: %.2f" %estimator.score(XLon_train, YLon_train) 
#print "Test R-squared: %.2f" %estimator.score(XLon_test, YLon_test) 


# In[53]:


#def GradientBoosterLat(param_grid, n_jobs):
    
#    estimator = GradientBoostingRegressor()
    
#    cv = ShuffleSplit(XLat_train.shape[0], test_size=0.2)
    
#    classifier = GridSearchCV(estimator=estimator,cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    
#    classifier.fit(XLat_train, YLat_train)
    
#    return cv, classifier.best_estimator_


# In[54]:


#param_grid={'n_estimators':[100], 
#            'learning_rate': [0.1, 0.05, 0.02, 0.01],
#            'max_depth':[6],
#            'min_samples_leaf':[3,5,9,17],
#            'max_features':[1.0,0.3,0.1]
#            }
#n_jobs=4
#cv,best_est=GradientBoosterLat(param_grid, n_jobs)


# In[55]:


#print "Best Estimator Parameters"

#print"---------------------------"
#print "n_estimators: %d" %best_est.n_estimators
#print "max_depth: %d" %best_est.max_depth
#print "Learning Rate: %.1f" %best_est.learning_rate
#print "min_samples_leaf: %d" %best_est.min_samples_leaf
#print "max_features: %.1f" %best_est.max_features
#print 
#print "Train R-squared: %.2f" %best_est.score(XLon_train,YLon_train)


# # STEP 4 – MODELLING

# ## Modelling Occupancy

# In[56]:


import datetime

def DateExtraction(row: Series):
    dt = datetime.datetime.fromtimestamp(row['Timestamp'])
    row['day'] = dt.weekday()
    row['hour'] = dt.hour
    row['minute'] = dt.minute
    row['monthday'] = dt.day
    return row


# In[57]:


data = pd.DataFrame()
for f in SelectedFiles:
    df = ConverttoDF(file_name=f)
    df = PreviousCoordinates(df=df)
    df = DistanceCalculation(df=df)
    data = pd.concat([data, df])


# In[58]:


data = data[['Latitude', 'Longitude', 'Occupancy', 'Timestamp']]
data['day'] = 0
data['hour'] = 0
data['minute'] = 0
data['monthday'] = 0
data = data.apply(lambda row: DateExtraction(row), axis=1)
data.head()


# In[59]:


X = data.drop(labels=['Occupancy', 'Timestamp'], axis=1)
X['Latitude'] = X['Latitude'] - lat_offset
X['Longitude'] = X['Longitude'] - long_offset

Y = data['Occupancy']


# In[60]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=0)


# #### Baseline Model

# I select a naive model that predicts always 0 for the next location. 
# 
# Naive model For naïve forecasts, we simply set all forecasts to be the value of the last observation.This method works remarkably well for many economic and financial time series.

# In[61]:


baseline_acc = round(Y_test[Y_test == 0].shape[0] / Y_test.shape[0], 3)
print(f'The baseline accuracy of a naive model is {baseline_acc}')


# #### Random Forest Model 

# I also build Random Forests in order to predict if a datapoint is a pick-up location or not. RF outperforms the baseline approach significantly.
# 
# Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.

# In[62]:


clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=0)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc = round(accuracy_score(Y_test, Y_pred), 3)
print(f'The accuracy is {acc}')


# ##### Feature Importance 

# We can also inspect and interpret the trained Random Forest classifier by analyzing the importance of each feature. Coordindates are the most important features to classify a pick-up point whereas the day feature does not help the classifier.

# In[63]:


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), [X_train.columns.tolist()[idx]  for idx in indices])
plt.xlim([-1, X_train.shape[1]])
plt.show()


# ### Adding new variable - Holiday Data

# I think also adding holiday and weather data will be very useful to our model. So I added holiday data as a flag and retrain Random Forest Classifier model. There is only one holiday in the given period of time which is Memorial Day(26th of May) .Result of both models are same but precision of predicting hailing a cab is increase 1%. Therefore we can use holiday flag for our model and it gives other data such as weather will be valuable.

# In[64]:


from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2008-01-01', end='2008-12-31').to_pydatetime()
print(holidays)


# In[65]:


data_new = data
data_new['Holiday_Flag'] = [(lambda x: (x in holidays) * 1)(x) for x in data_new['Timestamp']]


# In[66]:


print(data_new)


# In[67]:


X_new = data.drop(labels=['Occupancy', 'Timestamp'], axis=1)
X_new['Latitude'] = X_new['Latitude'] - lat_offset
X_new['Longitude'] = X_new['Longitude'] - long_offset

Y_new = data_new['Occupancy']


# In[68]:


X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X_new, Y_new, stratify=Y, test_size=0.2, random_state=0)


# In[69]:


baseline_acc_new = round(Y_new_test[Y_new_test == 0].shape[0] / Y_new_test.shape[0], 3)
print(f'The baseline accuracy of a naive model is {baseline_acc_new}')


# In[70]:


clf_new = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=0)
clf_new.fit(X_new_train, Y_new_train)
Y_pred_new = clf_new.predict(X_new_test)
acc_new = round(accuracy_score(Y_new_test, Y_pred_new), 3)
print(f'The accuracy is {acc_new}')


# In[71]:


importances = clf_new.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_new.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_new.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_new_train.shape[1]), [X_new_train.columns.tolist()[idx]  for idx in indices])
plt.xlim([-1, X_new_train.shape[1]])
plt.show()


# ### Model Comparisons 

# In[72]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
 
results = confusion_matrix(Y_test, Y_pred)
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(Y_test, Y_pred) )
print ('Report : ')
print (classification_report(Y_test, Y_pred) )


# I use accuracy as a metric, since the problem seems to be balanced (55% of the labels are 0 and 45% of the labels are 1). In a scenario with higher imbalance I could have used F1 score. So formulas and meanings for accuracy, F1 score, etc. below
# 
# 
# Accuracy: The proportion of the total number of predictions that were correct.
# 
# Positive Predictive Value (Precision): The proportion of positive cases correctly identified.
# 
# Negative Predictive Value: The proportion of negative cases correctly identified.
# 
# Sensitivity (Recall): The proportion of actual positive cases correctly identified.
# 
# Specificity: The proportion of actual negative cases correctly identified.
# 
# F1 Score= (2 * Precision * Recall) / (Precision + Recall)
# 
# Kappa = (Observed Accuracy -Expected Accuracy) / (1 -Expected Accuracy)

# In[73]:


results_new = confusion_matrix(Y_new_test, Y_pred_new) 
print ('Confusion Matrix :')
print(results_new) 
print ('Accuracy Score :',accuracy_score(Y_new_test, Y_pred_new) )
print ('Report : ')
print (classification_report(Y_new_test, Y_pred_new) )


# # STEP 6 – DEPLOYMENT

# Deployment is where data mining pays off. In this final phase of the Cross-Industry Standard Process for Data Mining (CRISP-DM) process, it doesn’t matter how brilliant your discoveries may be, or how perfectly your models fit the data, if you don’t actually use those things to improve the way that you do business.
# 
# The deployment phase includes four tasks. These are;
# 
# Planning deployment (your methods for integrating data-mining discoveries into use)
# 
# Planning monitoring and maintenance
# 
# Reporting final results
# 
# Reviewing final results

# # Bonus Question

# I apply two different method to identify cluster of taxi cabs.
# 
# First is using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) that is a popular unsupervised learning method utilized in model building and machine learning algorithms. I subset 10% of the total data and divided into two in terms of occupancy flag. I looked other cabs where they are driving in the past and where they will go after it. By doing so, I clustered its cabs together based on similarity of behavior.
# 
# My second method is implementing RFM methodology in to this question. I subset 10% of the total data and I found total miles per cab, total occupied miles per cab and average active minutes per day. Using mindset of RFM, I split the data into 4 in terms of quantile. Using this quantiles, I create 7 sample of segments. I gave the definition of segments in the coding phase.

# ## DBSCAN 

# In[74]:


def visualize_dbscan(db: DBSCAN, X: DataFrame):

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            continue

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask].values
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask].values
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


# In[75]:


sampled_data = data[data['Occupancy'] == 1][['Latitude', 'Longitude']].sample(frac=0.3, random_state=0)

db = DBSCAN(eps=0.0005, min_samples=50)
db.fit(sampled_data)
visualize_dbscan(db=db, X=sampled_data)


# In[76]:


sampled_data = data[data['Occupancy'] == 0][['Latitude', 'Longitude']].sample(frac=0.3, random_state=0)

db = DBSCAN(eps=0.0005, min_samples=50)
db.fit(sampled_data)
visualize_dbscan(db=db, X=sampled_data)


# ## Businesswise Segmentation (RFM Segmentation)

# When I think the data that I have, total miles per cab, total occupied miles per cab and average active minutes per day variable are more important than others. I extract those data below and create new dataset for RFM mindset segmentation

# In[77]:


import datetime

def DateExtraction(row: Series):
    dt = datetime.datetime.fromtimestamp(row['Timestamp'])
    row['day'] = dt.weekday()
    row['hour'] = dt.hour
    row['minute'] = dt.minute
    row['monthday'] = dt.day
    return row


# In[78]:


data = pd.DataFrame()
for f in SelectedFiles:
    df = ConverttoDF(file_name=f)
    df = PreviousCoordinates(df=df)
    df = DistanceCalculation(df=df)
    data = pd.concat([data, df])
data = data[['Latitude', 'Longitude', 'Occupancy', 'Taxi','Timestamp']]
data['day'] = 0
data['hour'] = 0
data['minute'] = 0
data['monthday'] = 0
data = data.apply(lambda row: DateExtraction(row), axis=1)
data.head()


# In[79]:


Segment = pd.DataFrame()
for f in SelectedFiles:
    df = ConverttoDF(file_name=f)
    df = PreviousCoordinates(df=df)
    df = DistanceCalculation(df=df)
    Segment = pd.concat([Segment, df])
    


# In[80]:


MileCover=pd.DataFrame()

MileCover = Segment.groupby(by=['Taxi'])['Miles'].sum()

print(MileCover)


# In[81]:


WPassangerDistance = Segment[Segment['Occupancy'] == 1]
#Distance Without Passanger
DistanceWPassanger=pd.DataFrame()
DistanceWPassanger = WPassangerDistance.groupby(by=['Taxi'])['Miles'].sum()


# In[82]:


RFMData = pd.merge(MileCover,DistanceWPassanger, on='Taxi', how='left')
print(RFMData)


# In[83]:



output  = pd.DataFrame()
output = data[['Taxi','monthday']]
output = output.drop_duplicates()
ActDay = pd.DataFrame()
ActDay = output.groupby(by=['Taxi'])['monthday'].count()


# In[84]:



#Active Minutes
ActMin = pd.DataFrame()
ActMin = data.groupby(by=['Taxi'])['Timestamp'].count()
#Active Days
output  = pd.DataFrame()
output = data[['Taxi','monthday']]
output = output.drop_duplicates()
ActDay = pd.DataFrame()
ActDay = output.groupby(by=['Taxi'])['monthday'].count()

Active = pd.DataFrame()
Active = pd.merge(ActMin, ActDay, on='Taxi', how='left')
#Finding Active Minutes per Day
Active['ActMinPerDay'] = Active['Timestamp'] / Active['monthday']


# In[85]:


RFMData = pd.merge(RFMData,Active, on='Taxi', how='left')
RFMDataF = RFMData[['Miles_x', 'Miles_y', 'ActMinPerDay']]
RFMDataF.rename(columns={'Miles_x': 'MileCoverage', 'Miles_y': 'MileCoverageWPassanger'}, inplace=True)
print(RFMDataF)


# After merging data, I found the quantiles value. This will help me to divide my data into four in terms of every variable such as Mile Coverages with Passanger.

# In[86]:


quantiles = RFMDataF.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
print(quantiles)


# In[87]:


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[88]:


RFMDataF['r_quartile'] = RFMDataF['MileCoverage'].apply(RScore, args=('MileCoverage',quantiles,))
RFMDataF['f_quartile'] = RFMDataF['MileCoverageWPassanger'].apply(FMScore, args=('MileCoverageWPassanger',quantiles,))
RFMDataF['m_quartile'] = RFMDataF['ActMinPerDay'].apply(FMScore, args=('ActMinPerDay',quantiles,))
RFMDataF.head()


# In[89]:


RFMDataF['RFMScore'] = RFMDataF.r_quartile.map(str) + RFMDataF.f_quartile.map(str) + RFMDataF.m_quartile.map(str)
RFMDataF.head()


# In[90]:


RFMDataF.groupby(by=['RFMScore'])['RFMScore'].count()


# So we can give clusters a name accordingly. Here is some example of it. 
# ---
# Best Taxi Drivers: Actively driving cab and working much more than other per day
# 
#      (if mile coverage, mile coverage with passanger and active minutes per day are in the top %25)
# ---               
# Globetrotter : Actively driving cab
# 
#      (if mile coverage is in the top %25)
# ---               
# Most Occupied Taxi : Actively driving cab with passanger
# 
#      (if mile coverage with passanger is in the top %25)
# ---               
# Most Active Taxi: Working much more than other per day
# 
#      (if active minutes per day is in the top %25)
# ---               
# Fast and Vacant: Actively driving more miles but less minutes per day. That means he/she drives fast with more vacant 
# 
#      (if mile coverage is in the top %25 and mile coverage with passanger and active minutes per day are in the bottom %25)
# ---               
# Lucky Stand-by: Driver is not driving without passanger but he/she drive more than other in terms of mile coverage with passanger and active minutes per day.
# 
#      (if mile coverage is in the bottom %25 and mile coverage with passanger and active minutes per day are in the top %25)
# ---                
# Not Active Taxi Drivers: Not actively driving cab and working much more than other per day
# 
#      (if mile coverage, mile coverage with passanger and active minutes per day are in the bottom %25)
# --- 

# In[91]:


print("Best Taxi Drivers: ",len(RFMDataF[RFMDataF['RFMScore']=='444']))
print('Globetrotter : ',len(RFMDataF[RFMDataF['r_quartile']==4]))
print('Most Occupied Taxi : ',len(RFMDataF[RFMDataF['f_quartile']==4]))
print("Most Active Taxi: ",len(RFMDataF[RFMDataF['m_quartile']==4]))
print('Fast and Vacant: ', len(RFMDataF[RFMDataF['RFMScore']=='411']))
print('Lucky Stand-by: ',len(RFMDataF[RFMDataF['RFMScore']=='144']))
print('Not Active Taxi Drivers: ',len(RFMDataF[RFMDataF['RFMScore']=='111']))

