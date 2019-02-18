import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

train = pd.read_csv('train.csv', delimiter=",")
test = pd.read_csv('test.csv', delimiter=",")

'''
SHAPE OF DATA SET
train = 614 x 13
test = 367 x 12
'''

# Filling missing values with 'Mean'
train['Credit_History'].fillna(train['Credit_History'].mean(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(), inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)

# Filling missing values with 'Forward Fill'
train['Gender'].fillna(method='ffill', inplace=True)
train['Married'].fillna(method='ffill', inplace=True)
train['Self_Employed'].fillna(method='ffill', inplace=True)
train['Dependents'].fillna(method='ffill', inplace=True)

# Dropping Loan_ID feature as it doesn't contribute much
train.drop('Loan_ID', axis=1, inplace=True)

print(train.columns)



for i in train.columns:
    if train[i].dtype == 'O':
        print(set(train[i].tolist()))






