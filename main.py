import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Lambda
from keras.losses import Huber
from keras.optimizers import Optimizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import datetime
import numpy as np
import pyupbit

from predict import get_dataframe, scale_dataframe, df_train_test_split, windowed_dataset, create_model, learning_data, \
    show_history

# upbit = pyupbit.Upbit(os.environ.get("UPBIT_ACCESS_KEY"), os.environ.get("UPBIT_SECRET_KEY"))
filename = os.path.join(".", "checkpoint.ckpt")

df = get_dataframe(interval="minute1", market="KRW-ZIL")
df = scale_dataframe(df)
x_train, x_test, y_train, y_test = df_train_test_split(df)

train_data = windowed_dataset(x_train, y_train, True)
test_data = windowed_dataset(x_test, y_test, False)
model = create_model()
history = learning_data(model=model, train_data=train_data, test_data=test_data, filename=filename)

show_history(model=model, history=history, test_data=test_data, y_test=y_test, filename=filename)
