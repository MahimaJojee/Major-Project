# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Screen3.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Screen3(object):
    def setupUi(self, Screen3_Window):
        Screen3_Window.setObjectName("Screen3_Window")
        Screen3_Window.resize(704, 483)
        self.centralwidget = QtWidgets.QWidget(Screen3_Window)
        self.centralwidget.setObjectName("centralwidget")
        self.Screen3Image = QtWidgets.QLabel(self.centralwidget)
        self.Screen3Image.setGeometry(QtCore.QRect(10, 50, 701, 461))
        self.Screen3Image.setText("")
        self.Screen3Image.setPixmap(QtGui.QPixmap("Images/metro10.jpg"))
        self.Screen3Image.setScaledContents(True)
        self.Screen3Image.setObjectName("Screen3Image")
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(0, 0, 701, 51))
        font = QtGui.QFont()
        font.setFamily("Rockwell Nova")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label3.setFont(font)
        self.label3.setAutoFillBackground(True)
        self.label3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label3.setIndent(70)
        self.label3.setObjectName("label3")
        self.label_3_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_3_1.setGeometry(QtCore.QRect(30, 90, 231, 51))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_3_1.setFont(font)
        self.label_3_1.setAutoFillBackground(True)
        self.label_3_1.setIndent(13)
        self.label_3_1.setObjectName("label_3_1")
        self.label_3_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_3_2.setGeometry(QtCore.QRect(400, 90, 251, 51))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_3_2.setFont(font)
        self.label_3_2.setAutoFillBackground(True)
        self.label_3_2.setIndent(13)
        self.label_3_2.setObjectName("label_3_2")
        self.DataVisualizeCombo1 = QtWidgets.QComboBox(self.centralwidget)
        self.DataVisualizeCombo1.setGeometry(QtCore.QRect(30, 190, 271, 51))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.DataVisualizeCombo1.setFont(font)
        self.DataVisualizeCombo1.setMaxVisibleItems(12)
        self.DataVisualizeCombo1.setObjectName("DataVisualizeCombo1")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo1.addItem("")
        self.DataVisualizeCombo2 = QtWidgets.QComboBox(self.centralwidget)
        self.DataVisualizeCombo2.setGeometry(QtCore.QRect(400, 190, 271, 51))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.DataVisualizeCombo2.setFont(font)
        self.DataVisualizeCombo2.setMaxVisibleItems(12)
        self.DataVisualizeCombo2.setObjectName("DataVisualizeCombo2")
        self.DataVisualizeCombo2.addItem("")
        self.DataVisualizeCombo2.addItem("")
        self.DataVisualizeCombo2.addItem("")
        self.DataVisualizeCombo2.addItem("")
        self.DataVisualizeCombo2.addItem("")
        self.DataVisualizeCombo2.addItem("")
        self.DataVisualizeCombo2.addItem("")
        self.Screen3_SubmitBtn1 = QtWidgets.QPushButton(self.centralwidget)
        self.Screen3_SubmitBtn1.setGeometry(QtCore.QRect(120, 240, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.Screen3_SubmitBtn1.setFont(font)
        self.Screen3_SubmitBtn1.setMouseTracking(True)
        self.Screen3_SubmitBtn1.setAutoFillBackground(True)
        self.Screen3_SubmitBtn1.setAutoDefault(False)
        self.Screen3_SubmitBtn1.setDefault(False)
        self.Screen3_SubmitBtn1.setFlat(False)
        self.Screen3_SubmitBtn1.setObjectName("Screen3_SubmitBtn1")
        self.Screen3_SubmitBtn2 = QtWidgets.QPushButton(self.centralwidget)
        self.Screen3_SubmitBtn2.setGeometry(QtCore.QRect(490, 240, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.Screen3_SubmitBtn2.setFont(font)
        self.Screen3_SubmitBtn2.setMouseTracking(True)
        self.Screen3_SubmitBtn2.setAutoFillBackground(True)
        self.Screen3_SubmitBtn2.setAutoDefault(False)
        self.Screen3_SubmitBtn2.setDefault(False)
        self.Screen3_SubmitBtn2.setFlat(False)
        self.Screen3_SubmitBtn2.setObjectName("Screen3_SubmitBtn2")
        self.BackButton = QtWidgets.QPushButton(self.centralwidget)
        self.BackButton.setGeometry(QtCore.QRect(260, 370, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.BackButton.setFont(font)
        self.BackButton.setMouseTracking(True)
        self.BackButton.setAutoFillBackground(True)
        self.BackButton.setAutoDefault(False)
        self.BackButton.setDefault(False)
        self.BackButton.setFlat(False)
        self.BackButton.setObjectName("BackButton")
        '''
        Screen3_Window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Screen3_Window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 704, 21))
        self.menubar.setObjectName("menubar")
        Screen3_Window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Screen3_Window)
        self.statusbar.setObjectName("statusbar")
        Screen3_Window.setStatusBar(self.statusbar)
        '''
        self.retranslateUi(Screen3_Window)
        QtCore.QMetaObject.connectSlotsByName(Screen3_Window)

    def retranslateUi(self, Screen3_Window):
        _translate = QtCore.QCoreApplication.translate
        Screen3_Window.setWindowTitle(_translate("Screen3_Window", "Screen3_Window"))
        self.label3.setText(_translate("Screen3_Window", "TREND DATA VISUALIZATIONS"))
        self.label_3_1.setText(_translate("Screen3_Window", "SCATTERPLOTS"))
        self.label_3_2.setText(_translate("Screen3_Window", "EXCEL FILE"))
        self.DataVisualizeCombo1.setItemText(0, _translate("Screen3_Window", "Sunday"))
        self.DataVisualizeCombo1.setItemText(1, _translate("Screen3_Window", "Monday"))
        self.DataVisualizeCombo1.setItemText(2, _translate("Screen3_Window", "Tuesday"))
        self.DataVisualizeCombo1.setItemText(3, _translate("Screen3_Window", "Wednesday"))
        self.DataVisualizeCombo1.setItemText(4, _translate("Screen3_Window", "Thursday"))
        self.DataVisualizeCombo1.setItemText(5, _translate("Screen3_Window", "Friday"))
        self.DataVisualizeCombo1.setItemText(6, _translate("Screen3_Window", "Saturday"))
        self.DataVisualizeCombo1.setItemText(7, _translate("Screen3_Window", "Weekends"))
        self.DataVisualizeCombo1.setItemText(8, _translate("Screen3_Window", "Weekdays"))
        self.DataVisualizeCombo1.setItemText(9, _translate("Screen3_Window", "Holidays"))
        self.DataVisualizeCombo1.setItemText(10, _translate("Screen3_Window", "Percentage Distribution"))
        self.DataVisualizeCombo2.setItemText(0, _translate("Screen3_Window", "Sunday"))
        self.DataVisualizeCombo2.setItemText(1, _translate("Screen3_Window", "Monday"))
        self.DataVisualizeCombo2.setItemText(2, _translate("Screen3_Window", "Tuesday"))
        self.DataVisualizeCombo2.setItemText(3, _translate("Screen3_Window", "Wednesday"))
        self.DataVisualizeCombo2.setItemText(4, _translate("Screen3_Window", "Thursday"))
        self.DataVisualizeCombo2.setItemText(5, _translate("Screen3_Window", "Friday"))
        self.DataVisualizeCombo2.setItemText(6, _translate("Screen3_Window", "Saturday"))
        self.Screen3_SubmitBtn1.setText(_translate("Screen3_Window", "SUBMIT"))
        self.Screen3_SubmitBtn2.setText(_translate("Screen3_Window", "SUBMIT"))
        self.BackButton.setText(_translate("Screen3_Window", "BACK"))

