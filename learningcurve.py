import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def plot_learning_curve(X, y, estimator):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    # Uncomment these to know the shapes before and after splitting - useful for Checking
    '''
    print("Before splitting:")
    print("X: ", X.shape)
    print("y: ", y.shape)
    print("\nAfter splitting:")
    print("X train: ", X_train.shape)
    print("X val: ", X_val.shape)
    print("y train: ", y_train.shape)
    print("y val: ", y_val.shape)'''

    # Number of different training set sizes to run the model over
    training_set_sizes = np.linspace(5, len(X_train), 10, dtype=int)

    # Creating empty lists to store training and validation scores
    train_score = []
    val_score = []

    # Iterating over each set and calculating the 'Accuracy Score'
    for i in training_set_sizes:
        estimator.fit(X_train.loc[0:i, :], y_train.loc[0:i])
        training_accuracy = accuracy_score(y_train.loc[0:i], estimator.predict(X_train.loc[0:i, :]))
        val_accuracy = accuracy_score(y_val, estimator.predict(X_val))
        train_score.append(training_accuracy)
        val_score.append(val_accuracy)


    # Plotting the figure in the graph
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(training_set_sizes, train_score, c='red')
    ax.plot(training_set_sizes, val_score, c='blue')
    fig.suptitle("Learning Curve", fontweight='bold', fontsize='16')
    ax.set_title("Estimator: Logistic Regression | Score: Accuracy_score", size=12)
    ax.set_xlabel("No. of training examples", size=16)
    ax.set_ylabel("Score", size=16)
    ax.legend(['Training set', 'Validation set'], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0, 1)

    plt.show()



