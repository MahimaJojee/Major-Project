# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Screen1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(693, 508)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Screen1_image = QtWidgets.QLabel(self.centralwidget)
        self.Screen1_image.setGeometry(QtCore.QRect(0, 130, 691, 421))
        self.Screen1_image.setText("")
        self.Screen1_image.setPixmap(QtGui.QPixmap("Images/metro4.jpg"))
        self.Screen1_image.setScaledContents(True)
        self.Screen1_image.setObjectName("Screen1_image")
        self.WelcomeLabel = QtWidgets.QLabel(self.centralwidget)
        self.WelcomeLabel.setGeometry(QtCore.QRect(0, 0, 691, 71))
        font = QtGui.QFont()
        font.setFamily("Rockwell Nova")
        font.setPointSize(26)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.WelcomeLabel.setFont(font)
        self.WelcomeLabel.setAutoFillBackground(True)
        self.WelcomeLabel.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.WelcomeLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.WelcomeLabel.setLineWidth(2)
        self.WelcomeLabel.setMidLineWidth(1)
        self.WelcomeLabel.setScaledContents(False)
        self.WelcomeLabel.setIndent(21)
        self.WelcomeLabel.setObjectName("WelcomeLabel")
        self.AgeAndGenderBtn = QtWidgets.QPushButton(self.centralwidget)
        self.AgeAndGenderBtn.setGeometry(QtCore.QRect(330, 350, 351, 61))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.AgeAndGenderBtn.setFont(font)
        self.AgeAndGenderBtn.setAutoFillBackground(True)
        self.AgeAndGenderBtn.setFlat(True)
        self.AgeAndGenderBtn.setObjectName("AgeAndGenderBtn")
        self.WelcomeLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.WelcomeLabel_2.setGeometry(QtCore.QRect(0, 70, 691, 61))
        font = QtGui.QFont()
        font.setFamily("Rockwell Nova")
        font.setPointSize(26)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.WelcomeLabel_2.setFont(font)
        self.WelcomeLabel_2.setAutoFillBackground(True)
        self.WelcomeLabel_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.WelcomeLabel_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.WelcomeLabel_2.setLineWidth(4)
        self.WelcomeLabel_2.setMidLineWidth(7)
        self.WelcomeLabel_2.setScaledContents(False)
        self.WelcomeLabel_2.setIndent(160)
        self.WelcomeLabel_2.setObjectName("WelcomeLabel_2")
        self.PredictionAnalysisBtn = QtWidgets.QPushButton(self.centralwidget)
        self.PredictionAnalysisBtn.setGeometry(QtCore.QRect(330, 170, 331, 61))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.PredictionAnalysisBtn.setFont(font)
        self.PredictionAnalysisBtn.setAutoFillBackground(True)
        self.PredictionAnalysisBtn.setFlat(True)
        self.PredictionAnalysisBtn.setObjectName("PredictionAnalysisBtn")
        self.TrendAnalysisBtn = QtWidgets.QPushButton(self.centralwidget)
        self.TrendAnalysisBtn.setGeometry(QtCore.QRect(330, 260, 351, 61))
        font = QtGui.QFont()
        font.setFamily("Sitka Small")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.TrendAnalysisBtn.setFont(font)
        self.TrendAnalysisBtn.setAutoFillBackground(True)
        self.TrendAnalysisBtn.setFlat(True)
        self.TrendAnalysisBtn.setObjectName("TrendAnalysisBtn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 693, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.WelcomeLabel.setText(_translate("MainWindow", "WELCOME TO TREND ANALYSIS "))
        self.AgeAndGenderBtn.setText(_translate("MainWindow", "AGE AND GENDER ANALYSIS"))
        self.WelcomeLabel_2.setText(_translate("MainWindow", "IN KOCHI METRO"))
        self.PredictionAnalysisBtn.setText(_translate("MainWindow", "PREDICTION ANALYSIS"))
        self.TrendAnalysisBtn.setText(_translate("MainWindow", "PASSENGER TREND ANALYSIS"))


