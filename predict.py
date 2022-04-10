import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Lambda
from keras.losses import Huber
from keras.optimizer_v2 import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import datetime
import numpy as np

import pyupbit


def get_dataframe(market: str, interval: str) -> DataFrame:
    df = pyupbit.get_ohlcv(ticker=market, interval=interval, count=10000)
    return df


def scale_dataframe(df: DataFrame) -> DataFrame:
    scaler = MinMaxScaler()
    scale_cols = ['open', 'high', 'low', 'close', 'volume', 'value']
    scaled = scaler.fit_transform(df[scale_cols])
    df = pd.DataFrame(scaled, columns=scale_cols)
    return df


def df_train_test_split(df: DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns='close'), df['close'], test_size=0.1,
                                                        random_state=0, shuffle=False)
    return x_train, x_test, y_train, y_test


def windowed_dataset(x, y, shuffle, window_size=5, batch_size=32):
    # X값 window dataset 구성
    ds_x = tf.data.Dataset.from_tensor_slices(x)
    ds_x = ds_x.window(window_size, shift=1, stride=1, drop_remainder=True)
    ds_x = ds_x.flat_map(lambda x: x.batch(window_size))
    # y값 추가
    ds_y = tf.data.Dataset.from_tensor_slices(y[window_size:])
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(batch_size).prefetch(1)


def create_model(window_size=5):
    # 1차원 feature map 생성
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[window_size, 5]),
        LSTM(16, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    loss = Huber()
    optimizer = adam(0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=["mse"])
    return model


def learning_data(model, train_data, test_data, filename):
    early_stopping = EarlyStopping(monitor='val_loss', patience=35)
    checkpoint = ModelCheckpoint(filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
    history = model.fit(train_data, validation_data=test_data, epochs=1000, callbacks=[checkpoint, early_stopping])
    return history

def show_history(model, history, test_data, y_test, filename, window_size=5):
    print("%.7f" % (float(min(history.history['val_loss']))))
    model.load_weights(filename)
    pred = model.predict(test_data)
    actual = np.asarray(y_test)[window_size:]
    actual = np.reshape(actual, (len(actual), 1))
    print(pred.shape)
    print(actual.shape)
    plt.figure(figsize=(10, 10))
    plt.plot(actual, label='actual')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()