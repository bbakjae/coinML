from math import radians
import os
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from prediction import Prediction
import pyupbit
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QDesktopWidget,
    QGridLayout,
    QListView,
    QRadioButton,
    QWidget,
    QPushButton,
    QTextEdit,
    QComboBox,
)
from PyQt5.QtCore import QCoreApplication, QDateTime, QRect
from PyQt5.QtGui import QStandardItemModel, QStandardItem


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
# filename = os.path.join(".", "checkpoint", "checkpoint.ckpt")
# market = "KRW-ZIL"
# interval = "minute1"

# scheduler = BackgroundScheduler()
# prediction = Prediction(market=market, interval=interval, filename=filename)

# scheduler.add_job(prediction.learning_data, "cron", hour=0)
# scheduler.start()

# while True:
#     df = prediction.get_live_dataframe()
#     pred = prediction.get_predict(df)
#     print(pred)
#     sleep(2)


# class MyApp(QWidget):
#     combo: QComboBox
#     list_view: QListView

#     def __init__(self) -> None:
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         grid = QGridLayout()
#         self.setLayout(grid)

#         self.combo = self.get_combobox()
#         self.list_view = self.get_list_view()

#         predict_btn = QPushButton("start predict", self)
#         learn_data_btn = QPushButton("start learning!", self)
#         learn_data_btn.clicked.connect(self.get_market_from_list)

#         text_edit = QTextEdit(self)
#         # text_edit.setUpdatesEnabled(False)

#         grid.addWidget(self.list_view, 0, 0)
#         grid.addWidget(self.combo, 0, 1)
#         grid.addWidget(learn_data_btn, 1, 0)
#         grid.addWidget(predict_btn, 1, 1)
#         grid.addWidget(text_edit, 2, 0, 1, 2)

#         datetime = QDateTime.currentDateTime().toString("yyyy.mm.dd hh:mm:ss")
#         # self.statusBar().showMessage(datetime)

#         self.setWindowTitle("Coin Prediction")
#         self.resize(300, 500)
#         self.center()
#         self.show()

#     def get_market_from_list(self):
#         index = self.list_view.currentIndex()
#         result = self.list_view.model().data(index)
#         return result

#     def learning_btn_action(self):
#         prediction = Prediction(
#             market=self.get_market_from_list,
#         )

#     def get_list_view(self) -> QListView:
#         list_view = QListView()
#         markets = pyupbit.get_tickers(fiat="KRW")
#         items = QStandardItemModel()
#         for i in markets:
#             items.appendRow(QStandardItem(i))
#         list_view.setModel(items)
#         return list_view

#     def get_combobox(self) -> QComboBox:
#         combo = QComboBox()
#         combo.addItem("1분봉")
#         combo.addItem("3분봉")
#         combo.addItem("5분봉")
#         combo.addItem("10분봉")
#         combo.addItem("30분봉")
#         combo.addItem("60분봉")
#         combo.addItem("240분봉")
#         combo.addItem("일봉")
#         combo.addItem("주봉")
#         combo.addItem("월봉")
#         return combo

#     def center(self):
#         qr = self.frameGeometry()
#         cp = QDesktopWidget().availableGeometry().center()
#         qr.moveCenter(cp)
#         self.move(qr.topLeft())


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ex = MyApp()
#     sys.exit(app.exec_())

