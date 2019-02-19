from sklearn.preprocessing import LabelEncoder


def prep_train_data(train):
    # Filling missing values with 'Mode'
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['Gender'].fillna(train['Gender'].mode(), inplace=True)
    train['Married'].fillna(train['Married'].mode(), inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode(), inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode(), inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

    # Filling missing Loan Amount values with 'Median'
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    # Dropping Loan_ID feature as it doesn't contribute much
    train.drop('Loan_ID', axis=1, inplace=True)


    # Encoding text to numerical values
    label_encoder = LabelEncoder()
    for i in train.columns:
        if train[i].dtype == 'O':
            train[i] = label_encoder.fit_transform(train[i])

    input_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                        'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    train_X = train[input_features]
    train_y = train['Loan_Status']



    return train_X, train_y

