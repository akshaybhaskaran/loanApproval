from sklearn.preprocessing import LabelEncoder
import numpy as np


def prep_test_data(test):
    # Filling missing values with 'Mean'
    test['Credit_History'].fillna(test['Credit_History'].mean(), inplace=True)
    test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(), inplace=True)
    test['LoanAmount'].fillna(test['LoanAmount'].mean(), inplace=True)

    # Filling missing values with 'Forward Fill'
    test['Gender'].fillna(method='ffill', inplace=True)
    test['Dependents'].fillna(method='ffill', inplace=True)
    test['Self_Employed'].fillna(method='ffill', inplace=True)

    # Dropping Loan_ID feature as it doesn't contribute much
    test.drop('Loan_ID', axis=1, inplace=True)

    # Encoding text to numerical values
    label_encoder = LabelEncoder()
    for i in test.columns:
        if test[i].dtype == 'O':
            test[i] = label_encoder.fit_transform(test[i])


    input_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                      'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    test_X = test[input_features]
    test_y = (np.zeros((len(test_X),), dtype=int))



    return test_X, test_y










