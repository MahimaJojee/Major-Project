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
from sklearn.preprocessing import LabelEncoder



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
            cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


csv_file =r'C:\Users\Mahima Jojee\Desktop\Copy of dupofCombinedSetMal.csv'

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H:%M:%S')
dataset = pd.read_csv(csv_file, parse_dates={'datetime': ['Date', 'Time']},date_parser=dateparse)
#dataset.drop('Unnamed: 0', axis=1, inplace=True)
dataset.columns = ['Date_time','Count','Location','Summary','Icon','Temperature','Humidity','Pressure','GroupNo','Day','DayNo','Is_Holiday']
dataset.to_csv(r'C:\Users\Mahima Jojee\Desktop\Copy of dupofCombinedSetMal2.csv')
dataset = pd.read_csv(r'C:\Users\Mahima Jojee\Desktop\Copy of dupofCombinedSetMal2.csv',index_col=0)
dataset.set_index('Date_time',inplace = True)
dataset.index = pd.to_datetime(dataset.index)
print(dataset.dtypes)
#print(type(dataset.index))


#DATA PREPARATION
values = dataset.iloc[:, [0,2,3 ,4, 5, 6, 7, 9, 10]].values
# 0:'Temperature',1:'Humidity',2:'Pressure',3:'Count',4:'GroupNo',5:'DayNo',6:'Is_Holiday'
encoder = LabelEncoder()
values[:, 1] = encoder.fit_transform(values[:, 1])
values[:, 2] = encoder.fit_transform(values[:, 2])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised l)earning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[-8,-7,-6,-5,-4,-3,-2,-1]], axis=1, inplace=True)
#print(reframed.head(5))


# split into input and outputs
values = reframed.values
train= values[:1000, : ]
test=values[1000: , : ]
train_X, train_y = train[:,:-1 ], train[:,-1]
test_X, test_y = test[:, :-1 ], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(100, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(LSTM(100 , return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(100 ,input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=800, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# evaluate the model
scores = model.evaluate(train_X, train_y, verbose=0)
print("model=",model.metrics_names)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("scores=",scores)
# save model and architecture to single file
model.save("model3.h5")
print("Saved model to disk")


