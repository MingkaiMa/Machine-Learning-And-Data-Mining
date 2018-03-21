#code for question 2 plot
import sys

import arff, numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import random

paths = ['balance-scale','primary-tumor',
         'glass','heart-h']
scores = [[0.35106382978723405, 0.76063829787234039, 0.73404255319148937, 0.6063829787234043, 0.19148936170212766], [0.30392156862745096, 0.47058823529411764, 0.36274509803921567, 0.35294117647058826, 0.049019607843137254], [0.36923076923076925, 0.69230769230769229, 0.66153846153846152, 0.47692307692307695, 0.12307692307692308], [0.5955056179775281, 0.7640449438202247, 0.7752808988764045, 0.38202247191011235, 0.30337078651685395]]
params = [[2, 2, 17, 17, 7], [2, 12, 7, 7, 22], [2, 7, 22, 12, 27], [2, 22, 22, 27, 17]]

header = "{:^123}".format("Decision Tree Results") + '\n' + '-' * 123  + '\n' + \
"{:^15} | {:^16} | {:^16} | {:^16} | {:^16} | {:^16} |".format("Dataset", "Default", "0%", "20%", "50%", "80%")


# print result table
print(header)
for i in range(len(scores)):
    #scores = score_list[i][1]
    print("{:<16}".format(paths[i]), end='')
    for j in range(len(params[i])):
        print("| {:>6.2%} ({:>2})     " .format(scores[i][j],params[i][j]), end='')
    print('|')
print('\n')
