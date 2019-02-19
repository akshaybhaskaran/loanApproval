import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from learningcurve import *
from trainData import *
from testData import *

train = pd.read_csv('train.csv', delimiter=",")
test = pd.read_csv('test.csv', delimiter=",")
submit = pd.read_csv('submission.csv')

submit['Loan_ID'] = test['Loan_ID']

# Code for preparing training data
train_X, train_y = prep_train_data(train)

# Code for preparing test data
test_X, test_y = prep_test_data(test)


model = LogisticRegression()
model.fit(train_X, train_y)
test_y = model.predict(test_X)

submit['Loan_Status'] = test_y
submit.to_csv("submit-1.csv", index=False)






#estimator = LogisticRegression()


#plot_learning_curve(train_X, train_y, estimator)


