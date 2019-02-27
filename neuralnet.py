from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)


def build_neuralNet(train_X, train_Y, test_X):

    nn_model = Sequential()
    nn_model.add(Dense(11, input_dim=11, activation='relu'))
    nn_model.add(Dense(15, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))

    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    nn_model.fit(train_X, train_Y, epochs=150, batch_size=10)

    scores = nn_model.evaluate(train_X, train_Y)
    print("\n%s: %.2f%%" %(nn_model.metrics_names[1], scores[1]*100))




