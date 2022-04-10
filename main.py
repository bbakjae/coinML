import os
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from prediction import Prediction


"""
 -- interval --
1분:       minute1
3분:       minute3
5분:       minute5
10분:      minute10
30분:      minute30
60분:      minute60
240분:     minute240
일봉:      day
주봉:      week
월봉:      month
"""
filename = os.path.join(".", "checkpoint", "checkpoint.ckpt")
market = "KRW-ZIL"
interval = "minute1"

scheduler = BackgroundScheduler()
prediction = Prediction(market=market, interval=interval, filename=filename)

scheduler.add_job(prediction.learning_data, "cron", hour=0)
scheduler.start()

while True:
    df = prediction.get_live_dataframe()
    pred = prediction.get_predict(df)
    print(pred)
    sleep(2)
