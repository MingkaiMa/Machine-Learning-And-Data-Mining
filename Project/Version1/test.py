import numpy as np
from knn_implementation import KNN_Classification, KNN_Regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

def testWithSklearn(k):
    neigh = KNeighborsClassifier(k)

    L = cross_val_score(neigh, self.X, self.Y, scoring = 'accuracy',
                        cv = KFold(n_splits=len(self.X)))

    return np.mean(L)

print('Test the implementation based on sklearn KNeighborsClassifier')

print('Test KNN classification, without weighted distance')

print('My solution       sklearn')

myList = []
skList = []

##for k in range(1, 20):
##    knn = KNN_Classification('./data/ionosphere.arff.txt', k)
##
##    myRes = knn.LOOCV()
##
##    neigh = KNeighborsClassifier(k)
##    L = cross_val_score(neigh, knn.X, knn.Y, scoring = 'accuracy',
##                        cv = KFold(n_splits=len(knn.X)))
##
##    skRes = np.mean(L)
##    myList.append(myRes)
##    skList.append(skRes)
##    print(f'{myRes:.12f}',end='')
##    print('   ', skRes)

##plt.plot(myList, label='My implement')
##plt.plot(skList, label='sklearn')
##plt.xlabel('k')
##plt.ylabel('accuracy')
##plt.legend()
##plt.show()




print('\nTest KNN classification, with weighted distance')
print('My solution       sklearn')

for k in range(1, 20):
    knn = KNN_Classification('./data/ionosphere.arff.txt', k)

    myRes = knn.LOOCV_weight_distance()

    neigh = KNeighborsClassifier(k, weights='distance')
    L = cross_val_score(neigh, knn.X, knn.Y, scoring = 'accuracy',
                        cv = KFold(n_splits=len(knn.X)))

    skRes = np.mean(L)
    myList.append(myRes)
    skList.append(skRes)
    print(f'{myRes:.12f}',end='')
    print('   ', skRes)

plt.plot(myList, label='My implement')
plt.plot(skList, label='sklearn')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend()
plt.show()

import sys
sys.exit()

print('\nTest KNN regression, without weighted distance')
print('My solution       sklearn')

for i in range(1, 20):
    knn = KNN_Regression('./data/autos.arff.txt', i)

    myRes = knn.LOOCV_weight_distance()
 

    print(f'{myRes:.12f}',end='\n')


