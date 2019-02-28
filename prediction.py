import pandas as pd
from learningcurve import *
from trainData import *
from testData import *
from neuralnet import *
from logisticRegression import *
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', delimiter=",")
test = pd.read_csv('test.csv', delimiter=",")
submit = pd.read_csv('sample_submission1.csv')
submit['Loan_ID'] = test['Loan_ID']

# Code for preparing Train and Test data
train_X, train_y = prep_train_data(train)
test_X = prep_test_data(test)

'''Shapes
train_X:  [614 x 11]
train_y:  [614]
test_X:   [367 x 11]
'''






# X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=0)

#lr_prediction = build_logisticRegression(train_X, train_y, test_X)
#nn_prediction = build_neuralNet(train_X, train_y, test_X)




#submit['Loan_Status'] = nn_prediction
#submit['Loan_Status'] = np.where(submit.Loan_Status >= 0.5, "Y", "N")
#submit.to_csv("sample_submission2.csv", index=False)








