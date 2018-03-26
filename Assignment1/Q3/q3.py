# code for question 3
import sys

import arff,numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics

#--------------Show the attributes--------------

dataset = arff.load(open('houses.arff',"r",encoding = "ISO-8859-1")) 
attributes = np.array(dataset['attributes'])

regr = linear_model.LinearRegression()
data = np.array(dataset['data'])
houses_X = data[:,1:] #X vector
houses_Y = data[:,0] #Y vector

regr.fit(houses_X, houses_Y)
intercept = regr.intercept_

print('---------default--------')
print('Intercept:\n%.2e' % intercept,end='\n')
print('Coefficients:')
for coef in regr.coef_:
    print('%.2e' % coef,end=" ")


file = open('q3.out','a')
file.write('Intercept:\n%.2e\n' % intercept)
file.write('Coefficients:\n')
for coef in regr.coef_:
    file.write('%.2e' % coef)
    file.write(' ')
file.write('\n')


predicted = cross_val_predict(regr, houses_X, houses_Y, cv=10)
RMSE = np.sqrt(metrics.mean_squared_error(houses_Y, predicted))
print ('\nRMSE:\n%.2e\n' % RMSE)
file.write('RMSE:\n%.2e\n' % RMSE)



#--------------Data transformation--------------

## log transform

regr = linear_model.LinearRegression()
data = np.array(dataset['data'])
houses_X = data[:,1:] #X vector
houses_Y = data[:,0] #Y vector

houses_Y = np.log(houses_Y)


regr.fit(houses_X, houses_Y)
intercept = regr.intercept_

print('\n---------log--------')
print('Intercept:\n%.2e' % intercept,end='\n')
print('Coefficients:')
for coef in regr.coef_:
    print('%.2e' % coef,end=" ")


##file = open('q3.out','a')
file.write('Intercept:\n%.2e\n' % intercept)
file.write('Coefficients:\n')
for coef in regr.coef_:
    file.write('%.2e' % coef)
    file.write(' ')
file.write('\n')


predicted = cross_val_predict(regr, houses_X, houses_Y, cv=10)
RMSE = np.sqrt(metrics.mean_squared_error(houses_Y, predicted))
print ('\nRMSE:\n%.2e\n' % RMSE)
file.write('RMSE:\n%.2e\n' % RMSE)



## squares transform

regr = linear_model.LinearRegression()
data = np.array(dataset['data'])
houses_X = data[:,1:] #X vector
houses_Y = data[:,0] #Y vector

houses_Y = houses_Y ** 2



regr.fit(houses_X, houses_Y)
intercept = regr.intercept_


print('\n---------square--------')

print('Intercept:\n%.2e' % intercept,end='\n')
print('Coefficients:')
for coef in regr.coef_:
    print('%.2e' % coef,end=" ")


##file = open('q3.out','a')
file.write('Intercept:\n%.2e\n' % intercept)
file.write('Coefficients:\n')
for coef in regr.coef_:
    file.write('%.2e' % coef)
    file.write(' ')
file.write('\n')


predicted = cross_val_predict(regr, houses_X, houses_Y, cv=10)
RMSE = np.sqrt(metrics.mean_squared_error(houses_Y, predicted))
print ('\nRMSE:\n%.2e\n' % RMSE)
file.write('RMSE:\n%.2e\n' % RMSE)

file.close()
