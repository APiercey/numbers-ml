import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model():
    model = Sequential([
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='linear'),
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    return model

def train_model(model, train_data):
    model.fit(
        train_data.loc[:, train_data.columns != 'label'],
        train_data['label'],
        epochs=10
    )

def predict(model, data):
    data = np.array([data])
    prediction = model.predict(data)
    return np.argmax(prediction)

def predict_and_compare(model, test_data, example_number):
    example = test_data.to_numpy()[example_number]
    real_value_of_y = example[0]
    example_values    = example[1:]
    prediction = predict(model, example_values)
    print(f"Prediction made: {prediction}. Real value: {real_value_of_y}")

train_data = pd.read_csv('./mnist_train.csv')
test_data = pd.read_csv('./mnist_test.csv')
