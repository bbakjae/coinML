import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D
from keras.losses import Huber
from keras.optimizer_v2 import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

import pyupbit


def create_model(window_size=5) -> Sequential:
    # 1차원 feature map 생성
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[window_size, 5]),
        LSTM(16, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    loss = Huber()
    optimizer = adam.Adam(0.0005)
    model.compile(loss=loss, optimizer=optimizer, metrics=["mse"])
    return model


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


class Prediction():
    scaler: MinMaxScaler
    model: Sequential
    market: str
    interval: str
    filename: str

    def __init__(self, market, interval, filename):
        self.market = market
        self.interval = interval
        self.filename = filename
        self.model = create_model()
        self.scaler = MinMaxScaler()

    def get_dataframe(self) -> DataFrame:
        df = pyupbit.get_ohlcv(ticker=self.market, interval=self.interval, count=10000)
        return df

    def scale_dataframe(self, df: DataFrame) -> DataFrame:
        scale_cols = ['open', 'high', 'low', 'close', 'volume', 'value']
        scaled = self.scaler.fit_transform(df[scale_cols])
        df = pd.DataFrame(scaled, columns=scale_cols)
        return df

    def get_learned_data(self, train_data, test_data):
        early_stopping = EarlyStopping(monitor='val_loss', patience=35)
        checkpoint = ModelCheckpoint(self.filename, save_weights_only=True, save_best_only=True, monitor='val_loss',
                                     verbose=1)
        history = self.model.fit(train_data, validation_data=test_data, epochs=1000,
                                 callbacks=[checkpoint, early_stopping])
        return history

    def show_history(self, history, test_data, y_test, window_size=5):
        print("%.7f" % (float(min(history.history['val_loss']))))
        self.model.load_weights(self.filename)
        pred = self.model.predict(test_data)
        actual = np.asarray(y_test)[window_size:]
        actual = np.reshape(actual, (len(actual), 1))
        plt.figure(figsize=(10, 10))
        plt.plot(actual, label='actual')
        plt.plot(pred, label='prediction')
        plt.legend()
        plt.show()

    def get_live_dataframe(self) -> DataFrame:
        df = self.get_dataframe()
        scaled_df, scaler = self.scale_dataframe(df=df)
        live_x, live_y = scaled_df.drop(columns="close"), scaled_df["close"]
        live_df = windowed_dataset(live_x, live_y, False)
        return live_df

    def get_predict(self, df: DataFrame, filename):
        self.model.load_weights(filename)
        prediction = self.model.predict(df)
        empty_df = np.zeros(shape=(len(prediction), 6))
        empty_df[:, 0] = prediction[:, 0]
        prediction = self.scaler.inverse_transform(empty_df)[:, 0]
        return prediction[-1]

    def learning_data(self):
        df = self.get_dataframe()
        df = self.scale_dataframe(df)
        x_train, x_test, y_train, y_test = df_train_test_split(df)
        train_data = windowed_dataset(x_train, y_train, True)
        test_data = windowed_dataset(x_test, y_test, False)
        history = self.get_learned_data(train_data=train_data, test_data=test_data)
        self.show_history(history=history, test_data=test_data, y_test=y_test)
