import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
import DateTime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from pandas import concat
from sklearn.metrics import mean_squared_error
from numpy import concatenate
import xlsxwriter
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


class ModelTraining(object):

    def convert_series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        no_of_features = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        columns, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            columns.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(no_of_features)]
            #print("i=",i,"names=",names)
            #print('columns=',columns)
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            columns.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(no_of_features)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(no_of_features)]
        # put it all together
        agg = concat(columns, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    def FramingInput(self):
        global test_dataset
        csv_file = r'../Input/InputDataset.csv'

        dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H:%M:%S')
        dataset = pd.read_csv(csv_file, parse_dates={'datetime': ['Date', 'Time']}, date_parser=dateparse)
        # dataset.drop('Unnamed: 0', axis=1, inplace=True)
        dataset.columns = ['Date_time', 'Count', 'Location', 'Summary', 'Icon', 'Temperature', 'Humidity', 'Pressure',
                           'GroupNo', 'Day', 'DayNo', 'Is_Holiday']
        dataset.to_csv(r'../Input/InputDataset.csv')
        dataset = pd.read_csv(r'../Input/InputDataset2.csv', index_col=0)
        test_dataset = dataset.iloc[1000:, :]
        dataset.set_index('Date_time', inplace=True)
        dataset.index = pd.to_datetime(dataset.index)
        print(dataset.dtypes)

        # DATA PREPARATION
        values = dataset.iloc[:, [0,2,3 ,4, 5, 6, 7, 9, 10]].values  # 0:'Temperature',1:'Humidity',2:'Pressure',3:'Count',4:'GroupNo',5:'DayNo',6:'Is_Holiday'
        encoder = LabelEncoder()
        values[:, 1] = encoder.fit_transform(values[:, 1])
        values[:, 2] = encoder.fit_transform(values[:, 2])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)
        # frame as supervised learning
        self.reframed = self.convert_series_to_supervised(scaled, 1, 1)
        self.reframed.drop(self.reframed.columns[[-8,-7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)
        self.values = self.reframed.values

    def Prediction(self):
        global p_data,inv_y,inv_yhat,history,plot_data
        # split into input and outputs
        values = self.reframed.values
        train = self.values[:1000, :]
        test = self.values[1000:, :]
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # load model which has already been trained with 800 epochs
        model = load_model('model3.h5')
        # summarize model.
        model.summary()

        # evaluate the model
        score = model.evaluate(train_X, train_y, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        # make a prediction
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        plot_data = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = plot_data[:, 0]

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]
        p_data = plot_data.astype(int)
       

    def Workbook_write(self):

        # writing predicted and actual value to excel sheet
        workbook = xlsxwriter.Workbook('PredictionOutput.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'ActualCount')
        worksheet.write('B1', 'PredictedCount')
        worksheet.write('C1', 'Difference')
        row = 1

        for i in range(inv_yhat.size):
            testing = inv_y[i].astype('int32')
            worksheet.write(row, 0, testing)
            predicting = inv_yhat[i].astype('int32')
            worksheet.write(row, 1, predicting)
            worksheet.write(row, 2, testing - predicting)
            row += 1
        workbook.close()

    # plot history
    def Epochs_Plot(self):
        pyplot.plot(history.history['loss'], label='Training Loss')
        pyplot.plot(history.history['val_loss'], label='Validation Loss')
        pyplot.title("Training and Validation Loss")
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Loss")
        pyplot.legend()
        pyplot.show()

        # Plotting histogram of Difference
    def Histogram(self):
        diff_plot = pd.read_excel(r"../Output/Saturday_Count.xlsx")
        ax = diff_plot.hist(column='Difference')
        ax = ax[0]
        for x in ax:
            x.set_title("Histogram showing difference between actual and predicted passenger count")
            x.set_xlabel("Magnitude of Difference", labelpad=20, weight='bold', size=12)
            x.set_ylabel("Number of Records", labelpad=20, weight='bold', size=12)
        pyplot.show()

        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

    # Scatter plot of actual and predicted value
    def Comparision_scatterplot(self):
        x1 = np.arange(1, 407)
        y1 = inv_yhat
        y2 = inv_y
        pyplot.scatter(x1, y1, color='red', label="Predicted Value")
        pyplot.scatter(x1, y2, color='blue', label="Actual Value")
        pyplot.title("PREDICTION OUTPUT")
        pyplot.xlabel("INPUT")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()



 # Weekwise plotting of count

    def Sunday(self):
        n = p_data[:, 0][p_data[:, -2] == 6].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 6], color='black', label='SUNDAY COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON SUNDAY")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()


    def Monday(self):
        n = p_data[:, 0][p_data[:, -2] == 0].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 0], color='red', label='MONDAY COUNT')
        # pyplot.pie(x=np.arange(0,n),y=p_data[ : ,0][p_data[: ,-2]==0],autopct = "%.2f")
        # pyplot.axes().set_aspect("equal")
        pyplot.title(" PASSENGER CROWD EXPECTED ON MONDAY")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()

    def Tuesday(self):
        n = p_data[:, 0][p_data[:, -2] == 1].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 1], color='black', label='TUESDAY COUNT')
        pyplot.title(" PASSENGER CROWD EXPECTED ON TUESDAY")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()

    def Wednesday(self):
        n = p_data[:, 0][p_data[:, -2] == 2].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 2], color='blue', label='WEDNESDAY COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON WEDNESDAY")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()

    def Thursday(self):
        n = p_data[:, 0][p_data[:, -2] == 3].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 3], color='purple', label='THURSDAY COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON THURSDAY ")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()

    def Friday(self):
        n = p_data[:, 0][p_data[:, -2] == 4].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 4], color='red', label='FRIDAY COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON FRIDAY")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()

    def Saturday(self):
        n = p_data[:, 0][p_data[:, -2] == 5].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -2] == 5], color='green', label='SATURDAY COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON SATURDAY")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()


    def Weekends(self):
        n = p_data[:, 0][(p_data[:, -2] == 5) ^ (p_data[:, -2] == 6).any()].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][(p_data[:, -2] == 5) ^ (p_data[:, -2] == 6).any()],
                       color='purple', label='WEEKEND COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON WEEKENDS(SAT & SUN) ")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()


    def Weekdays(self):
        n = p_data[:, 0][
            (p_data[:, -2] == 0) ^ (p_data[:, -2] == 1) ^ (p_data[:, -2] == 2) ^ (
                    p_data[:, -2] == 3) ^ (
                    p_data[:, -2] == 4).any()].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][
            (p_data[:, -2] == 0) ^ (p_data[:, -2] == 1) ^ (p_data[:, -2] == 2) ^ (
                    p_data[:, -2] == 3) ^ (
                    p_data[:, -2] == 4).any()], color='purple', label='WEEKDAYS COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON WEEKDAYS COUNT ")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()


    def Holidays(self):
        n = p_data[:, 0][p_data[:, -1] == 1].shape[0]
        pyplot.scatter(np.arange(0, n), p_data[:, 0][p_data[:, -1] == 1], color='black',
                       label='HOLIDAY COUNT')
        pyplot.title("PASSENGER CROWD EXPECTED ON HOLIDAYS")
        pyplot.xlabel("NUMBER OF RECORDS")
        pyplot.ylabel("PASSENGER COUNT VALUE")
        pyplot.legend(loc='best')
        pyplot.show()


        # Converting output numpy array to DataFrame
    def Conversion_To_DataFrame(self):
        global df
        framed_Predicted_Data = {'Count': p_data[:, 0], 'GroupNo': p_data[:, -3], 'DayNo': p_data[:, -2],
                                 'Is_Holiday': p_data[:, -1]}
        # framed_Predicted_Data={'Count' : [p_data[:][0]],'GroupNo':[p_data[:][1]],'DayNo':[p_data[:][-3]],'Is_Holiday':[p_data[:][3]]}
        df = pd.DataFrame(framed_Predicted_Data, columns=['Count', 'GroupNo', 'DayNo', 'Is_Holiday'])
        

 # Passenger Distribution by Days of the Week
    def Percentage_Distribution(self):
        my_labels = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        my_explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        df.Count.groupby(df.DayNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%', explode=my_explode)
        pyplot.axis('equal')
        pyplot.title("Passenger Distribution by Weekdays")
        pyplot.show()

        # Passenger GROUP[AGE AND GENDER] Distribution by Days of the Week
    def pd_Monday(self):
        my_labels = '1-15 Male','1-15 Female','16-30 Male','16-30 Female','31-50 Male','31-50 Female','51-70 Male','51-70 Female','70 above Male','70 above Female'
        my_explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        df.GroupNo[df.DayNo == 0].groupby(df.GroupNo).sum().plot(kind='pie',labels=my_labels,autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Monday")
        pyplot.show()

    def pd_Tuesday(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[df.DayNo == 1].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Tuesday")
        pyplot.show()

    def pd_Wednesday(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        #my_explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
        df.GroupNo[df.DayNo == 2].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Wednesday")
        pyplot.show()

    def pd_Thursday(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[df.DayNo == 3].groupby(df.GroupNo).sum().plot(kind='pie',labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Thursday")
        pyplot.show()

    def pd_Friday(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[df.DayNo == 4].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Friday")
        pyplot.show()

    def pd_Saturday(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[df.DayNo == 5].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Saturday")
        pyplot.show()

    def pd_Sunday(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[df.DayNo == 6].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Sunday")
        pyplot.show()

    def pd_Weekends(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[(df.DayNo == 5) ^ (df.DayNo == 6)].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Weekends")
        pyplot.show()

    def pd_Weekdays(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[(df.DayNo == 0) ^ (df.DayNo == 1) ^ (df.DayNo == 2) ^ (df.DayNo == 3) ^ (df.DayNo == 4)].groupby(
            df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Weekdays")
        pyplot.show()

    def pd_Holidays(self):
        my_labels = '1-15 Male', '1-15 Female', '16-30 Male', '16-30 Female', '31-50 Male', '31-50 Female', '51-70 Male', '51-70 Female', '70 above Male', '70 above Female'
        df.GroupNo[df.Is_Holiday == 1].groupby(df.GroupNo).sum().plot(kind='pie', labels=my_labels, autopct='%1.1f%%')
        pyplot.axis('equal')
        pyplot.title("Passenger Age and Gender distribution on Holidays")
        pyplot.show()



    def excel_Sunday(self):
        workbook = xlsxwriter.Workbook('Sunday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 6].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 6]
        new_df = test_dataset[test_dataset['DayNo'] == 6]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()

    def excel_Monday(self):
        workbook = xlsxwriter.Workbook('Monday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 0].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 0]
        new_df = test_dataset[test_dataset['DayNo'] == 0]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()

    def excel_Tuesday(self):
        workbook = xlsxwriter.Workbook('Tuesday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 1].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 1]
        new_df = test_dataset[test_dataset['DayNo'] == 1]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()


    def excel_Wednesday(self):
        workbook = xlsxwriter.Workbook('Wednesday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 2].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 2]
        new_df = test_dataset[test_dataset['DayNo'] == 2]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()


    def excel_Thursday(self):
        workbook = xlsxwriter.Workbook('Thursday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 3].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 3]
        new_df = test_dataset[test_dataset['DayNo'] == 3]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()



    def excel_Friday(self):
        workbook = xlsxwriter.Workbook('Friday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 4].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 4]
        new_df = test_dataset[test_dataset['DayNo'] == 4]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()


    def excel_Saturday(self):
        workbook = xlsxwriter.Workbook('Saturday_Count.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Date & Time')
        worksheet.write('B1', 'Expected Count')
        worksheet.write('C1', 'GroupNo')
        row = 1
        n = p_data[:, 0][p_data[:, -2] == 5].shape[0]
        new_array = p_data[:, :][p_data[:, -2] == 5]
        new_df = test_dataset[test_dataset['DayNo'] == 5]
        for i in range(n):
            datetime = new_df.iloc[i, 0]
            worksheet.write(row, 0, datetime)
            count = new_array[i, 0]
            worksheet.write(row, 1, count)
            group_no = new_df.iloc[i, -4]
            worksheet.write(row, 2, group_no)
            row += 1
        workbook.close()




def BackEnd_Main():
    trainer=ModelTraining()
    trainer.FramingInput()
    trainer.Prediction()
    trainer.Workbook_write()



