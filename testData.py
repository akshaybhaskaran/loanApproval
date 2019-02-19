from sklearn.preprocessing import LabelEncoder



def prep_test_data(test):
    # Filling missing values with 'Mode'
    test['Gender'].fillna(test['Gender'].mode(), inplace=True)
    test['Dependents'].fillna(test['Dependents'].mode(), inplace=True)
    test['Self_Employed'].fillna(test['Self_Employed'].mode(), inplace=True)
    test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
    test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)

    # Filling missing Loan Amount values with 'Median'
    test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

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






    return test_X










