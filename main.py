import numpy as np
import random
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


def create_data():
    x = []
    y = []
    for i in range(10000):
        x.append(i)
        y.append(i % 2)
    x = np.array(x)
    y = np.array(y)
    np.savez('data', x=x, y=y)


def load_data(filename):
    npzfile = np.load(filename)
    x = npzfile['x']
    y = npzfile['y']
    return x, y

def main():
    x, y = load_data('data.npz')
    x, y = shuffle(x, y, random_state=6)
    x_train = x[:7000:]
    y_train = y[:7000:]
    x_valid = x[7000::]
    y_valid = y[7000::]

    print(len(x_train), len(x_valid))

    model = Sequential()
    model.add(Input(1))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(4, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=[x_valid, y_valid])


#create_data()
main()