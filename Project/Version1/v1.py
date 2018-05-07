import arff
import numpy as np
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



class KNN:
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
            


    
    def getKNNClassifierResult(self, test_data):
        res = []

        for i in range(len(self.X)):
            distance = self.euclideanDistance(self.X[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

##        print(final_k)
        class_dic = {}
        index_list = [i[0] for i in final_k]
        
        for i in index_list:
            if self.Y[i] not in class_dic:
                class_dic[self.Y[i]] = 1
            else:
                class_dic[self.Y[i]] += 1

##        for i in class_dic:
##            print(i, '  ', class_dic[i])
        
        dic = sorted(class_dic, key = lambda x:class_dic[x], reverse=True)
##        print(dic)



        
        return dic[0]



    def getKNNClassifierResult_LOOCV(self, test_data, leave_index):


        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))
        
        
        
        res = []

        for i in range(len(train_data)):
            distance = self.euclideanDistance(train_data[i], test_data)
            res.append((i, distance))


        res = sorted(res, key = lambda x: x[1])
        final_k = res[: self.k]

        #print(final_k)
        class_dic = {}
        index_list = [i[0] for i in final_k]
        
        for i in index_list:
            if train_label[i] not in class_dic:
                class_dic[train_label[i]] = 1
            else:
                class_dic[train_label[i]] += 1

##        for i in class_dic:
##            print(i, '  ', class_dic[i])
        dic = sorted(class_dic, key = lambda x:class_dic[x], reverse=True)
##        print(dic)

        if(len(dic) == 1):
            return dic[0]

        else:
            if(class_dic[dic[0]] == class_dic[dic[1]]):
                dic2 = sorted(dic)
                return dic2[0]
            else:
                return dic[0]
        
##        print(class_dic)
##        
##        return dic[0]

    #
    # Leave one out cross validation
    #

    def getKNNClassifierResult_LOOCV_sklearn(self, test_data, leave_index):
        train_data = np.concatenate((self.X[:leave_index], self.X[leave_index+1: ]))
        train_label = np.concatenate((self.Y[:leave_index], self.Y[leave_index+1: ]))
        
        neigh = KNeighborsClassifier(n_neighbors = self.k,
                                     algorithm='brute')


        neigh.fit(train_data, train_label)

        res = neigh.predict([test_data])[0]

        return res
        
        
        
    
    def LOOCV(self):
        correct_predict = 0
        total_predict = self.X.shape[0]

        

        i = 0

        while(i < total_predict):
            #print(i)
            res = self.getKNNClassifierResult_LOOCV(self.X[i], i)
            res2 = self.getKNNClassifierResult_LOOCV_sklearn(self.X[i], i)
            
            if res == self.Y[i]:
                correct_predict += 1

            if res != res2:
                print(f'self method result: {i}: {res}  sklearn: {res2}')

            
            i += 1

        print(f'correct predict: {correct_predict}')
        print(f'accuracy is: {correct_predict / total_predict}')

        
            

    def testWithSklearn(self):
        neigh = KNeighborsClassifier(n_neighbors = self.k)

        L = cross_val_score(neigh, self.X, self.Y, scoring = 'accuracy',
                            cv = KFold(n_splits=len(self.X)))

        nn = 0
        for i in L:
            if i == 1:
                nn += 1

        print(len(L))
        print(nn)
        print(np.mean(L))
        
        
        
        
        
                
                


        
# test:

##knn = KNN('./data/ionosphere.arff.txt', 10)
##
##test_instance = knn.X[4]
##res = knn.getKNNClassifierResult(test_instance)
##
##if res == knn.Y[4]:
##    print('yes')
##        

##knn = KNN('./data/ionosphere.arff.txt', 10)
##knn.LOOCV()

# log:
# accuracy is: 0.8347578347578347


for i in range(1, 20):
    knn = KNN('./data/ionosphere.arff.txt', i)
    knn.LOOCV()
##leave_index = 236
####leave_index = 250
##train_data = np.concatenate((knn.X[:leave_index], knn.X[leave_index+1: ]))
##train_label = np.concatenate((knn.Y[:leave_index], knn.Y[leave_index+1: ]))


##neigh = KNeighborsClassifier(n_neighbors = 10,
##                             algorithm='brute')
##
##neigh.fit(train_data, train_label)
##
##
##test_data = knn.X[leave_index]
##neigh.predict([test_data])
##neigh.predict_proba([test_data])
##neigh.kneighbors([test_data])


