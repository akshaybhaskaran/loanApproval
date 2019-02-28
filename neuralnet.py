from keras.models import Sequential
from keras.layers import Dense
import numpy as np


from sklearn.metrics import confusion_matrix
import numpy



def build_neuralNet(train_X, train_Y, test_X):
    numpy.random.seed(7)

    nn_model = Sequential()
    nn_model.add(Dense(15, activation='relu',
                       kernel_initializer='random_normal', input_dim=11))
    nn_model.add(Dense(15, activation='relu',
                       kernel_initializer='random_normal'))
    nn_model.add(Dense(15, activation='relu',
                       kernel_initializer='random_normal'))
    nn_model.add(Dense(1, activation='sigmoid',
                       kernel_initializer='random_normal'))

    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    nn_model.fit(train_X, train_Y, epochs=100, batch_size=10)

    scores = nn_model.evaluate(train_X, train_Y)
    print("\n%s: %.2f%%" %(nn_model.metrics_names[1], scores[1]*100))



    prediction = nn_model.predict(test_X)
    print(prediction)

    layer = nn_model.get_layer(index=2)
    print(layer.get_weights())



    #prediction = np.where(prediction >= 0.5, "Y", "N")
    #print(prediction.shape)









