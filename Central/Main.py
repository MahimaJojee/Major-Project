import numpy as np
import pandas as pd
from matplotlib import pyplot
import xlsxwriter
from sklearn.model_selection import learning_curve

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGridLayout, QWidget, QDesktopWidget


import BackEnd as back
import Screen1
import Screen2
import Screen3
import Screen4
import os


class OutputMappings(QtWidgets.QMainWindow,back.ModelTraining, Screen1.Ui_MainWindow):

    def __init__(self):
        print("At Central")
        super(OutputMappings,self).__init__()
        back.BackEnd_Main()
        self.setupUi(self)
        self.centerMain()

        print("Out of frontend")

        self.PredictionAnalysisBtn.clicked.connect(self.PathToScreen2)
        self.TrendAnalysisBtn.clicked.connect(self.PathToScreen3)
        self.AgeAndGenderBtn.clicked.connect(self.PathToScreen4)

    def centerMain(self):
        dimension = self.frameGeometry()
        centerPosition = QDesktopWidget().availableGeometry().center()
        dimension.moveCenter(centerPosition)
        self.move(dimension.topLeft())

    def PathToScreen2(self):
        self.window = QtWidgets.QWidget()
        self.Screen2_Object=Screen2.Ui_Screen2()
        self.Screen2_Object.setupUi(self.window)
        MainWindow.hide()
        self.window.show()
        self.Screen2_Object.Screen2_SubmitBtn.clicked.connect(self.Submission1)
        self.Screen2_Object.BackButton.clicked.connect(self.ReturnBack)


    def Submission1(self):
        #self.Screen2_Object = Screen2.Ui_Screen2()
        self.backend_object = back.ModelTraining()
        if (self.Screen2_Object.PredictedAndActualCombo.currentText() == "By number of Epochs"):
            self.backend_object.Epochs_Plot()
        if (self.Screen2_Object.PredictedAndActualCombo.currentText() == "Excel File of Output"):
            self.ExcelFile()
        if (self.Screen2_Object.PredictedAndActualCombo.currentText() == "Histogram showing Difference"):
            self.backend_object.Histogram()
        if (self.Screen2_Object.PredictedAndActualCombo.currentText() == "Scatter Plot of Output"):
            self.backend_object.Comparision_scatterplot()



    def PathToScreen3(self):
        self.window = QtWidgets.QWidget()
        self.Screen3_Object = Screen3.Ui_Screen3()
        self.Screen3_Object.setupUi(self.window)
        MainWindow.hide()
        self.window.show()
        self.Screen3_Object.Screen3_SubmitBtn1.clicked.connect(self.Submission2)
        self.Screen3_Object.Screen3_SubmitBtn2.clicked.connect(self.Submission3)
        self.Screen3_Object.BackButton.clicked.connect(self.ReturnBack)

    def Submission2(self):
        #self.Screen3_Object = Screen3.Ui_Screen3()
        self.backend_object = back.ModelTraining()
        self.backend_object = back.ModelTraining()
        if(self.Screen3_Object.DataVisualizeCombo1.currentText()=="Sunday"):
            self.backend_object.Sunday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Monday"):
            self.backend_object.Monday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Tuesday"):
            self.backend_object.Tuesday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Wednesday"):
            self.backend_object.Wednesday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Thursday"):
            self.backend_object.Thursday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Friday"):
            self.backend_object.Friday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Saturday"):
            self.backend_object.Saturday()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Weekends"):
            self.backend_object.Weekends()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Weekdays"):
            self.backend_object.Weekdays()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Holidays"):
            self.Holidays()
        if (self.Screen3_Object.DataVisualizeCombo1.currentText() == "Percentage Distribution"):
            self.backend_object.Conversion_To_DataFrame()
            self.backend_object.Percentage_Distribution()

    def Submission3(self):
        #self.Screen3_Object = Screen3.Ui_Screen3()
        self.backend_object = back.ModelTraining()
        self.backend_object = back.ModelTraining()
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Sunday"):
            self.backend_object.excel_Sunday()
            file = "../Output/Sunday_Count.xlsx"
            os.startfile(file)
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Monday"):
            self.backend_object.excel_Monday()
            file = "../Output/Monday_Count.xlsx"
            os.startfile(file)
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Tuesday"):
            self.backend_object.excel_Tuesday()
            file = "../Output/Tuesday_Count.xlsx"
            os.startfile(file)
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Wednesday"):
            self.backend_object.excel_Wednesday()
            file = "../Output/Wednesday_Count.xlsx"
            os.startfile(file)
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Thursday"):
            self.backend_object.excel_Thursday()
            file = "../Output/Thursday_Count.xlsx"
            os.startfile(file)
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Friday"):
            self.backend_object.excel_Friday()
            file = "../Output/Friday_Count.xlsx"
            os.startfile(file)
        if (self.Screen3_Object.DataVisualizeCombo2.currentText() == "Saturday"):
            self.backend_object.excel_Saturday()
            file = "../Output/Saturday_Count.xlsx"
            os.startfile(file)

    def PathToScreen4(self):
        self.window = QtWidgets.QWidget()
        self.Screen4_Object = Screen4.Ui_Screen4()
        self.Screen4_Object.setupUi(self.window)
        MainWindow.hide()
        self.window.show()
        self.Screen4_Object.Screen4_SubmitBtn.clicked.connect(self.Submission4)
        self.Screen4_Object.BackButton.clicked.connect(self.ReturnBack)


    def Submission4(self):
        #self.Screen4_Object = Screen4.Ui_Screen4()
        self.backend_object = back.ModelTraining()
        self.backend_object.Conversion_To_DataFrame()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Sunday"):
            self.backend_object.pd_Sunday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Monday"):
            self.backend_object.pd_Monday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Tuesday"):
            self.backend_object.pd_Tuesday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Wednesday"):
            self.backend_object.pd_Wednesday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Thursday"):
            self.backend_object.pd_Thursday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Friday"):
            self.backend_object.pd_Friday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Saturday"):
            self.backend_object.pd_Saturday()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Weekends"):
            self.backend_object.pd_Weekends()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Weekdays"):
            self.backend_object.pd_Weekdays()
        if (self.Screen4_Object.AgeGenderCombo.currentText() == "Holidays"):
            self.backend_object.pd_Holidays()

    def ExcelFile(self):
        #import os
        file = "../Output/PredictionOutput.xlsx"
        os.startfile(file)

    def ReturnBack(self):
        self.window.hide()
        MainWindow.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = OutputMappings()
    MainWindow.show()
    sys.exit(app.exec_())

