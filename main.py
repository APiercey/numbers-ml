import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model():
    model = Sequential([
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='linear'),
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    return model

def train_model(model, train_data, epochs=10):
    model.fit(
        train_data.loc[:, train_data.columns != 'label'],
        train_data['label'],
        epochs=epochs
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

def test(model, test_data):
    tmp_test_data = test_data.copy()
    example_data = tmp_test_data.loc[:, tmp_test_data.columns != 'label']
    predictions = model.predict(example_data)
    tmp_test_data['prediction'] = np.argmax(predictions,axis=1)
    tmp_test_data['correct_prediction'] = tmp_test_data['prediction'] == tmp_test_data['label']
    correct_percentage = (len(tmp_test_data[tmp_test_data['correct_prediction'] == True]) / len(tmp_test_data)) * 100
    print(f'Correctly guessed {correct_percentage}% of predictions.')


train_data = pd.read_parquet('./mnist_train.parquet')
test_data = pd.read_parquet('./mnist_test.parquet')

# model = build_model()
# train_model(model, train_data, epochs=100) # 94.96%
# test(model, test_data)
# predict_and_compare(model, test_data, 100)
