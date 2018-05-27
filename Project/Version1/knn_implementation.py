#
# implement knn classification and regression
#

import arff
import numpy as np
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class KNN_Classification:
    #
    # Read data from file and preprocess
    #
    def __init__(self, data_file, K):
        
        self.dataset = arff.load(open(data_file, 'r'))
        self.data = np.array(self.dataset['data'])

        self.X = np.array(self.data)[:, :-1]
        self.X = self.X.astype(np.float)
        self.Y = np.array(self.data)[:, -1]


        if(K > self.X.shape[0]):
            self.k = self.X.shape[0]
        else:
            self.k = K

    # Calculate euclidean distance between two instances
    def euclideanDistance(self, instance1, instance2):
        distance = 0

        for i in range(len(instance1)):
            distance += math.pow((instance1[i] - instance2[i]), 2)

        return math.sqrt(distance)

    

    #
    # Calculate 
    #
    def weighted_distance(sefl, instance1, instance2):
        distance = 0
        for i in range(len(instance1)):
            distance += math.pow((instance1[i] - instance2[i]), 2)

        if distance == 0:
            return 1
        return 1 / math.sqrt(distance)


    #
    # Predict result, where the test data is from outside, not from the training data,
    # this function is not used in LOOCV validation.
    #
    def getKNNClassifierResult(self, test_data):
        res = []

        for i in range(len(self.X)):
            distance = self.euclideanDistance(self.X[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

        class_dic = {}
        index_list = [i[0] for i in final_k]
        
        for i in index_list:
            if self.Y[i] not in class_dic:
                class_dic[self.Y[i]] = 1
            else:
                class_dic[self.Y[i]] += 1


        
        dic = sorted(class_dic, key = lambda x:class_dic[x], reverse=True)       
        return dic[0]

    #
    # Predict result, where the test data is from outside, not from the training data,
    # this function is not used in LOOCV validation.
    #
    def getKNNClassifierResult_weight_distance(self, test_data):
        res = []

        for i in range(len(self.X)):
            distance = self.euclideanDistance(self.X[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]
        #print(final_k)
        class_dic = {}
        index_list = [i[0] for i in final_k]
        
        for i in index_list:
            if self.Y[i] not in class_dic:
                class_dic[self.Y[i]] = self.weighted_distance(self.X[i], test_data)
            else:
                class_dic[self.Y[i]] += self.weighted_distance(self.X[i], test_data)


        
        dic = sorted(class_dic, key = lambda x:class_dic[x], reverse=True)       
        return dic[0]

    #
    # predict result, using leave one out cross validation
    #
    def getKNNClassifierResult_LOOCV(self, test_data, leave_index):


        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))
        
        res = []

        for i in range(len(train_data)):
            distance = self.euclideanDistance(train_data[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

        class_dic = {}
        index_list = [i[0] for i in final_k]
        
        for i in index_list:
            if train_label[i] not in class_dic:
                class_dic[train_label[i]] = 1
            else:
                class_dic[train_label[i]] += 1


        dic = sorted(class_dic, key = lambda x:class_dic[x], reverse=True)


        if(len(dic) == 1):
            return dic[0]

        else:
            #
            # when the top k has the same votes
            # the result is according to the sklearn KNeighborsClassifier
            #
            if(class_dic[dic[0]] == class_dic[dic[1]]):
                dic2 = sorted(dic)
                return dic2[0]
            else:
                return dic[0]


    #
    # predict result, using leave one out cross validation
    #
    def getKNNClassifierResult_LOOCV_weight_distance(self, test_data, leave_index):


        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))
        
        res = []

        for i in range(len(train_data)):
            distance = self.euclideanDistance(train_data[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

        class_dic = {}
        index_list = [i[0] for i in final_k]


        for i in index_list:
            if train_label[i] not in class_dic:
                class_dic[train_label[i]] = self.weighted_distance(train_data[i], test_data)
            else:
                class_dic[train_label[i]] += self.weighted_distance(train_data[i], test_data)



        dic = sorted(class_dic, key = lambda x:class_dic[x], reverse=True)

        return dic[0]

    #
    # leave one out cross validation
    #
    def LOOCV(self):
        correct_predict = 0
        total_predict = self.X.shape[0]

        i = 0

        while(i < total_predict):
            res = self.getKNNClassifierResult_LOOCV(self.X[i], i)
            
            if res == self.Y[i]:
                correct_predict += 1

            i += 1

        return correct_predict / total_predict

    #
    # leave one out cross validation
    #
    def LOOCV_weight_distance(self):
        correct_predict = 0
        total_predict = self.X.shape[0]

        i = 0

        while(i < total_predict):
            res = self.getKNNClassifierResult_LOOCV_weight_distance(self.X[i], i)
                
            if res == self.Y[i]:
                correct_predict += 1

            i += 1

        return correct_predict / total_predict



class KNN_Regression:

    #
    # Read data from file and preprocess
    #
    def __init__(self, data_file, K):

        self.dataset = arff.load(open(data_file, 'r'))
        self.data = np.array(self.dataset['data'])

        self.X = np.array(self.data)[:, :]

        no_missing_index = []

        for i in range(len(self.X)):
            l = list(self.X[i])
            if None in l:
                continue

            no_missing_index.append(i)

        continuous_index = []
        for i in range(len(self.dataset['attributes'])):
            if(self.dataset['attributes'][i][1] == 'REAL') or self.dataset['attributes'][i][0] == 'symboling':
                continuous_index.append(i)

        price_index = continuous_index.pop(-2)

        self.X = self.X[no_missing_index, :]
        self.X = self.X[:, continuous_index]
        for i in self.X:
            i[-1] = int(i[-1])
            
        self.Y = np.array(self.data)[:, price_index]
        self.Y = self.Y[no_missing_index]

        if(K > self.X.shape[0]):
            self.k = self.X.shape[0]
        else:
            self.k = K

    # Calculate euclidean distance between two instances
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
    

    def getKNNRegressorResult(self, test_data):
        res = []

        for i in range(len(self.X)):
            distance = self.euclideanDistance(self.X[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        
        final_k = res[: self.k]
        
        index_list = [i[0] for i in final_k]

        top_k_value = [self.Y[i] for i in index_list]
        return np.mean(top_k_value)


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
    

    def getKNNRegressorResult_LOOCV(self, test_data, leave_index):
        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))

        res = []

        for i in range(len(train_data)):
            distance = self.euclideanDistance(train_data[i], test_data)
            res.append((i, distance))        


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

        index_list = [i[0] for i in final_k]
        top_k_value = [train_label[i] for i in index_list]
        return np.mean(top_k_value)
    

    def getKNNRegressorResult_LOOCV_weight_distance(self, test_data, leave_index):

        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))

        res = []       

        for i in range(len(train_data) - 1, -1, -1):
            distance = self.euclideanDistance(train_data[i], test_data)
            res.append((i, distance)) 

        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

        index_list = [i[0] for i in final_k]

        res = 0
        total_weight = 0
        
        for index in index_list:
            own_weight = self.weighted_distance(train_data[index], test_data)
            total_weight += own_weight
            res += train_label[index] * own_weight

            if self.euclideanDistance(test_data, train_data[index]) == 0:
                return train_label[index]

        return res / total_weight

    def LOOCV(self):
        error_rate = []

        total_time = self.X.shape[0]

        i = 0
        while(i < total_time):
            res = self.getKNNRegressorResult_LOOCV(self.X[i], i)

            err = abs(res - self.Y[i]) / self.Y[i]
            error_rate.append(err)
            
            i += 1

        return np.mean(error_rate)

    def LOOCV_weight_distance(self):
        error_rate = []
        total_time = self.X.shape[0]

        i = 0
        while(i < total_time):

            res = self.getKNNRegressorResult_LOOCV_weight_distance(self.X[i], i)
            err = abs(res - self.Y[i]) / self.Y[i]
            error_rate.append(err)
            i += 1


        return np.mean(error_rate)
    
