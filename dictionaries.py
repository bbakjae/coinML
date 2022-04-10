import pyupbit

intervals = {
    "1분": "minute1",
    "3분": "minute3",
    "5분": "minute5",
    "10분": "minute10",
    "30분": "minute30",
    "60분": "minute60",
    "240분": "minute240",
    "일봉": "day",
    "주봉": "week",
    "월봉": "month",
}

markets = pyupbit.get_tickers(fiat="KRW")