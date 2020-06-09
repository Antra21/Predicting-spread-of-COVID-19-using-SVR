# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:18:05 2020

@author: antra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
import datetime
plt.style.use('seaborn')
confirmed=pd.read_csv(r'C:\Users\antra\Downloads\novel-corona-virus-2019-dataset (1)\time_series_covid_19_confirmed.csv')
recovered=pd.read_csv(r'C:\Users\antra\Downloads\novel-corona-virus-2019-dataset (1)\time_series_covid_19_recovered.csv')
deaths=pd.read_csv(r'C:\Users\antra\Downloads\novel-corona-virus-2019-dataset (1)\time_series_covid_19_deaths.csv')
confirmed=confirmed.iloc[:,4:108]
recovered=recovered.iloc[:,4:108]
deaths=deaths.iloc[:,4:108]

#creating array 1D,2D
cases=[]
mortality_rate=[]
total_deaths=[]
total_recovered=[]

#selecting the dates
dates=confirmed.keys()
for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recovered[i].sum()
    cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)
#append adds single item to the end of the given list

days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
cases=np.array(cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)
days_in_future=15
future_predict=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_predict[:-15]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_dates = []
for i in range(len(future_predict)):
    future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
## now applying train test
svc_model=SVC()
X_train,X_test,y_train,y_test=train_test_split(days_since_1_22,cases, test_size=0.15)
svc_model.fit(X_train  ,y_train)
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, cases, test_size=0.15, shuffle=False)
y = y_train_confirmed.ravel()
y_train_confirmed = np.array(y).astype(int)
kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}
svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

#for confirmed
# =============================================================================
svm_search.fit(X_train_confirmed, y_train_confirmed)
svm_search.best_params_
svm_confirmed = svm_search.best_estimator_
y_pred_confirmed = svm_confirmed.predict(X_test_confirmed)
# plt.plot(X_test_confirmed,y_pred_confirmed)
# plt.plot(X_test_confirmed,y_test_confirmed)

#for recovered
# =============================================================================
X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered = train_test_split(days_since_1_22, cases, test_size=0.15, shuffle=False)
y = y_train_recovered.ravel()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
svm_search.fit(X_train_recovered, y_train_recovered)
svm_search.best_params_
svm_recovered = svm_search.best_estimator_
y_pred_recovered = svm_recovered.predict(X_test_recovered)
# plt.plot(X_test_recovered,y_pred_recovered)
# plt.plot(X_test_recovered,y_test_recovered)


#for deaths
X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths = train_test_split(days_since_1_22, cases, test_size=0.15, shuffle=False)
y = y_train_deaths.ravel()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
svm_search.fit(X_train_deaths, y_train_deaths)
svm_search.best_params_
svm_deaths = svm_search.best_estimator_
y_pred_deaths = svm_deaths.predict(X_test_deaths)
#plt.plot(X_test_deaths,y_pred_deaths)
#plt.plot(X_test_deaths,y_test_deaths)

y_pred_future=svm_confirmed.predict(future_predict)
svm_pred=svm_confirmed.predict(future_predict)
svm_test_pred=svm_confirmed.predict(X_test_confirmed)
#plt.plot(future_predict,y_pred_future)

y_pred_future=svm_recovered.predict(future_predict)
svm_pred=svm_recovered.predict(future_predict)
svm_test_pred=svm_recovered.predict(X_test_recovered)
#plt.plot(future_predict,y_pred_future)


y_pred_future=svm_deaths.predict(future_predict)
svm_pred=svm_deaths.predict(future_predict)
svm_test_pred=svm_deaths.predict(X_test_deaths)
plt.plot(future_predict,y_pred_future)

y_pred_future=svm_confirmed.predict(future_predict)
svm_pred=svm_confirmed.predict(future_predict)
svm_test_pred=svm_confirmed.predict(X_test_confirmed)
#plt.plot(future_predict,y_pred_future)
