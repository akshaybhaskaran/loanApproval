from sklearn.linear_model import LogisticRegression


def build_logisticRegression(train_X, train_y, test_X):
    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)

    lr_model.fit(train_X, train_y)
    prediction = lr_model.predict(test_X)
    return prediction





