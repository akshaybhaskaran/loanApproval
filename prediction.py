import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

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


# Encoding text to numerical values
label_encoder = LabelEncoder()
for i in train.columns:
    if train[i].dtype == 'O':
        train[i] = label_encoder.fit_transform(train[i])

input_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

train_X = train[input_features]
train_y = train['Loan_Status']


'''
Next steps:
i. Split the model into train and CV sets
ii. Plot learning curves
iii. Based on the curves, work around the features and parameters
'''