import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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
test_X = prep_test_data(test)

# X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=0)

model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)

model.fit(train_X, train_y)
prediction = model.predict(test_X)
estimator = model
plot_learning_curve(train_X, train_y, estimator)



# submit['Loan_Status'] = prediction
# submit['Loan_Status'] = np.where(submit.Loan_Status == 1, "Y", "N")
# submit.to_csv("sample_submission1.csv", index=False)








