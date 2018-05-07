#
# implement knn regression
#

import arff
import numpy as np
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



data_file = './data/autos.arff.txt'
        
dataset = arff.load(open(data_file, 'r'))
data = np.array(dataset['data'])

X = np.array(data)[:, :-1]
Y = np.array(data)[:, -1]


        

 
