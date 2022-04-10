import os

from prediction import Prediction


filename = os.path.join(".", "checkpoint", "checkpoint.ckpt")
market = "KRW-ZIL"
interval = "minute1"

prediction = Prediction(market=market, interval=interval, filename=filename)
prediction.learning_data()
