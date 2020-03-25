#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]
dataset.dtypes
import datetime as dt
dataset['Date'] = pd.to_datetime(dataset.Date)
dataset['Date']=dataset['Date'].map(dt.datetime.toordinal)
#Splitting dataset intro training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()'''


#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting Test set Results
y_pred = regressor.predict(X_test)

#Visualising training set
plt.scatter(X_train, y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('year vs annual change')
plt.xlabel('year')
plt.ylabel('annual change')
plt.show()

#Visualising test set
plt.scatter(X_test, y_test, color ='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('year vs annual change')
plt.xlabel('year')
plt.ylabel('annual change')
plt.show()