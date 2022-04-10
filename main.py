import os

import predict
from predict import Prediction

filename = os.path.join(".", "checkpoint", "checkpoint.ckpt")
market = "KRW-ZIL"
interval = "minute1"

prediction: Prediction = predict.Prediction(market=market, interval=interval, filename=filename)
prediction.learning_data()
