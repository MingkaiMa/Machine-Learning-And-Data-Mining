#
# implement knn regression
#

import arff
import numpy as np
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



            

data_file = './data/autos.arff.txt'

class knnRegressor:
    def __init__(self, data_file, K):

        
        self.dataset = arff.load(open(data_file, 'r'))
        self.data = np.array(self.dataset['data'])

        self.X = np.array(self.data)[:, :-1]

        no_missing_index = []

        for i in range(len(self.X)):
            l = list(self.X[i])
            if None in l:
                continue

            no_missing_index.append(i)

        continuous_index = []
        for i in range(len(self.dataset['attributes'])):
            if(self.dataset['attributes'][i][1] == 'REAL'):
                continuous_index.append(i)


        price_index = continuous_index.pop()

        self.X = self.X[no_missing_index, :]
        self.X = self.X[:, continuous_index]
        self.Y = np.array(self.data)[:, price_index]
        self.Y = self.Y[no_missing_index]

        if(K > self.X.shape[0]):
            self.k = self.X.shape[0]
        else:
            self.k = K


    def euclideanDistance(self, instance1, instance2):
        distance = 0

        for i in range(len(instance1)):
            distance += math.pow((instance1[i] - instance2[i]), 2)

        return math.sqrt(distance)


    #
    # this is according to the sklearn KNeighborsClassifier(weights='distance')
    #
    def weighted_distance(self, instance1, instance2):
        distance = 0
        for i in range(len(instance1)):
            distance += math.pow((instance1[i] - instance2[i]), 2)

        if distance == 0:
            return 1
        return 1 / math.sqrt(distance)


    def getKNNRegressorResult_weighted_distance(self, test_data):
        res = []

        for i in range(len(self.X)):
            distance = self.euclideanDistance(self.X[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        
        final_k = res[: self.k]
 #       print(final_k)

        index_list = [i[0] for i in final_k]


        res = 0
        total_weight = 0
        
        for index in index_list:
            own_weight = self.weighted_distance(self.X[index], test_data)
            total_weight += own_weight
            res += self.Y[index] * own_weight


        return res / total_weight
            
        
## #       print(index_list)
##        top_k_value = [self.Y[i] for i in index_list]
## #       print(top_k_value)
##
##        return np.mean(top_k_value)


    def getKNNRegressorResult_LOOCV_weight_distance(self, test_data, leave_index):

        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))

        res = []

##        for i in range(len(train_data)):
##
##            distance = self.euclideanDistance(train_data[i], test_data)
##            res.append((i, distance))        

        for i in range(len(train_data) - 1, -1, -1):
            distance = self.euclideanDistance(train_data[i], test_data)
            res.append((i, distance)) 

        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]
        print(final_k)
        index_list = [i[0] for i in final_k]
##        print(index_list)
        res = 0
        total_weight = 0
        
        for index in index_list:
            own_weight = self.weighted_distance(train_data[index], test_data)
            total_weight += own_weight
            res += train_label[index] * own_weight

            if self.euclideanDistance(test_data, train_data[index]) == 0:
                return train_label[index]


##        print(res)
##        print(total_weight)
        return res / total_weight



        
##        return np.mean(top_k_value)


    def getKNNRegressorResult_LOOCV_sklearn(self, test_data, leave_index):
        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))
        

        neigh = KNeighborsRegressor(n_neighbors=self.k,
                                    weights='distance')

        neigh.fit(train_data, train_label)
        res = neigh.predict([test_data])

        return res

    

    def LOOCV(self):
        error_rate = []
        error_rate2 = []
        total_time = self.X.shape[0]

        i = 0
        while(i < total_time):

            res = self.getKNNRegressorResult_LOOCV_weight_distance(self.X[i], i)
            res2 = self.getKNNRegressorResult_LOOCV_sklearn(self.X[i], i)[0]

            if(res != res2):
                print(i, ' !==========')
                print(res, ' ', res2)

            #print(res, ' ', res2)
            err = abs(res - self.Y[i]) / self.Y[i]
            err2 = abs(res2 - self.Y[i]) / self.Y[i]

            error_rate.append(err)
            error_rate2.append(err2)

            i += 1

##        print(np.mean(error_rate))
##        print(np.mean(error_rate2))

        return np.mean(error_rate)


    def testWithSklearn(self):
        error_rate = []
        total_time = self.X.shape[0]

        i = 0
        while(i < total_time):

            res2 = self.getKNNRegressorResult_LOOCV_sklearn(self.X[i], i)

            err = abs(res2 - self.Y[i]) / self.Y[i]

            error_rate.append(err)

            i += 1

        return np.mean(error_rate)        
        

## not exactly same with sklearn, because there are some same distances points, sklearn may choose
## different points with my own choice
##    
##print('My solution       sklearn')
##
##for i in range(1, 20):
##    knn = knnRegressor(data_file, i)
##    print('=============', i)
##    myRes = knn.LOOCV()
##    skRes = knn.testWithSklearn()
##
##    print(f'{myRes:.12f}',end='')
##    print('   ', skRes)
##
##    
knn = knnRegressor(data_file, 1)
##print(knn.LOOCV())

##print(knn.testWithSklearn())
##data_file = './data/autos.arff.txt'
##knn = knnRegressor(data_file, 10)
##
##test_instance = knn.X[1]
##
##res = knn.getKNNRegressorResult(test_instance)
##
##neigh = KNeighborsRegressor(n_neighbors = knn.k)
##neigh.fit(knn.X, knn.Y)
##print(neigh.predict([test_instance]))


leave_index = 42
train_data = np.concatenate((knn.X[:leave_index], knn.X[leave_index+1: ]))
train_label = np.concatenate((knn.Y[:leave_index], knn.Y[leave_index+1: ]))

test_data = knn.X[leave_index]
neigh = KNeighborsRegressor(n_neighbors = knn.k, weights='distance')

neigh.fit(train_data, train_label)
neigh.predict([test_data])

