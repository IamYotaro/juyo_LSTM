import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

#%%
if os.path.isfile('juyo_info.pkl'):
    juyo_info = pd.read_pickle('juyo_info.pkl')
else:
    juyo_2018_info = pd.read_csv('http://www.tepco.co.jp/forecast/html/images/juyo-2018.csv', encoding="shift-jis")
    juyo_2018_info = juyo_2018_info[1:]
    juyo_2018_info.rename(columns={juyo_2018_info.columns.values[0]: 'Consumption'}, inplace=True)
    juyo_2018_info = juyo_2018_info['Consumption'].astype(int)
    juyo_2018_info = juyo_2018_info.reset_index()
    juyo_2018_info.rename(columns={'level_0': 'date'}, inplace=True)
    juyo_2018_info.rename(columns={'level_1': 'time'}, inplace=True)

    juyo_2017_info = pd.read_csv('http://www.tepco.co.jp/forecast/html/images/juyo-2017.csv', encoding="shift-jis")
    juyo_2017_info = juyo_2017_info[1:]
    juyo_2017_info.rename(columns={juyo_2017_info.columns.values[0]: 'Consumption'}, inplace=True)
    juyo_2017_info = juyo_2017_info['Consumption'].astype(int)
    juyo_2017_info = juyo_2017_info.reset_index()
    juyo_2017_info.rename(columns={'level_0': 'date'}, inplace=True)
    juyo_2017_info.rename(columns={'level_1': 'time'}, inplace=True)

    juyo_2016_info = pd.read_csv('http://www.tepco.co.jp/forecast/html/images/juyo-2016.csv', encoding="shift-jis")
    juyo_2016_info = juyo_2016_info[1:]
    juyo_2016_info.rename(columns={juyo_2016_info.columns.values[0]: 'Consumption'}, inplace=True)
    juyo_2016_info = juyo_2016_info['Consumption'].astype(int)
    juyo_2016_info = juyo_2016_info.reset_index()
    juyo_2016_info.rename(columns={'level_0': 'date'}, inplace=True)
    juyo_2016_info.rename(columns={'level_1': 'time'}, inplace=True)

    juyo_info = pd.concat([juyo_2016_info, juyo_2017_info, juyo_2018_info])
    
    juyo_info['datetime'] = juyo_info['date'] + ' ' + juyo_info['time']
    juyo_info.drop(['date', 'time'], axis=1, inplace=True)
    juyo_info = juyo_info.loc[:,['datetime', 'Consumption']]    
    juyo_info['datetime'] = pd.to_datetime(juyo_info['datetime'])
    juyo_info = juyo_info.set_index('datetime') 
    
    juyo_info.to_pickle('juyo_info.pkl')
    juyo_info.to_csv('juyo_info.csv')

#%%
juyo_info = juyo_info.set_index([juyo_info.index,
                                 juyo_info.index.year, 
                                 juyo_info.index.month, 
                                 juyo_info.index.weekday, 
                                 juyo_info.index.day, 
                                 juyo_info.index.hour])
juyo_info.index.names = ['datetime','year', 'month', 'weekday','day', 'hour']

#%%
# ex. Extraction Year(2017)
# print(juyo_info.xs(2017, level='Year'))
# ex. Extraction 2018-01-01 Weekly(Monday)
# print(juyo_info.loc[pd.IndexSlice[:, 2018, 1, :, 1],:])

juyo_201801to02_info = juyo_info.loc[pd.IndexSlice[:, 2018, 1:2],:]
juyo_201801to02_info.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
print(juyo_201801to02_info)

#%%
# plot juyo_info in 2018/01~02
fig, ax = plt.subplots(1,1)
ax.plot(juyo_201801to02_info[:'2018-01-31 23:00:00'])
ax.plot(juyo_201801to02_info['2018-02-01 00:00:00':])
ax.xaxis.set_major_locator(mdates.DayLocator([(i*5) for i in range(7)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_title('Daily Electric Power Consumption in 2018/01~02',fontsize=13)
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
fig.autofmt_xdate(rotation=90, ha='center')
plt.tight_layout()
plt.savefig("juyo_201801to02.png",dpi=300)
plt.show()

#%%
# plot juyo_info in 2018/01/01
juyo_20180101_info = juyo_info.loc[pd.IndexSlice[:, 2018, 1, :, 1], :]
juyo_20180101_info.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
print(juyo_20180101_info)

fig, ax = plt.subplots(1,1)
ax.plot(juyo_20180101_info)
ax.xaxis.set_major_locator(mdates.HourLocator([(i*2) for i in range(12)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator())
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
ax.set_title('Daily Electric Power Consumption in 2018/01/01',fontsize=13)
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
fig.autofmt_xdate(rotation=90, ha='center')
plt.tight_layout()
plt.savefig("juyo_20180101.png",dpi=300)
plt.show()

#%%
# make daily average juyo_info in 2018/01
juyo_201801_info = juyo_info.loc[pd.IndexSlice[:, 2018, 1],:]
juyo_201801_info.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
print(int(len(juyo_201801_info)/24+1))

juyo_201801_daily_ave = []
for i in range(1, int(len(juyo_201801_info)/24+1)):
    xday_temp = juyo_info.loc[pd.IndexSlice[:, 2018, 1, :, i], :]
    xday_temp.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
    xday_ave = xday_temp['Consumption'].mean()
    
    juyo_201801_daily_ave.append(xday_ave)

start_date = '2018-01-01'
end_date = '2018-01-31'
day_201801 = pd.date_range(start_date, end_date, freq='D')

juyo_201801_daily_ave = pd.Series(juyo_201801_daily_ave)
day_201801 = pd.Series(day_201801)
juyo_201801_daily_ave = pd.concat([day_201801, juyo_201801_daily_ave], axis=1)
juyo_201801_daily_ave = juyo_201801_daily_ave.set_index(0)
juyo_201801_daily_ave.rename(columns={1: 'Consumption'}, inplace=True)

#%%
# plot daily average juyo_info in 2018/01
fig, ax = plt.subplots(1,1)
ax.plot(juyo_201801_daily_ave)
ax.xaxis.set_major_locator(mdates.DayLocator([(i*2) for i in range(16)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
ax.set_title('Daily average Electirc Power Consumption in 2018/01',fontsize=13)
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
fig.autofmt_xdate(rotation=90, ha='center')
plt.tight_layout()
plt.savefig("juyo_201801_daily_ave.png",dpi=300)
plt.show()

#%%
# make daily average juyo_info in 2018/02
# plot daily average juyo_info in 2018/02
juyo_201802_info = juyo_info.loc[pd.IndexSlice[:, 2018, 2],:]
juyo_201802_info.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
print(int(len(juyo_201802_info)/24+1))

juyo_201802_daily_ave = []
for i in range(1, int(len(juyo_201802_info)/24 +1)):
    xday_temp = juyo_info.loc[pd.IndexSlice[:, 2018, 2, :, i], :]
    xday_temp.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
    xday_ave = xday_temp['Consumption'].mean()
    
    juyo_201802_daily_ave.append(xday_ave)

start_date = '2018-02-01'
end_date = '2018-02-28'
day_201802 = pd.date_range(start_date, end_date, freq='D')

juyo_201802_daily_ave = pd.Series(juyo_201802_daily_ave)
day_201802 = pd.Series(day_201802)
juyo_201802_daily_ave = pd.concat([day_201802, juyo_201802_daily_ave], axis=1)
juyo_201802_daily_ave = juyo_201802_daily_ave.set_index(0)
juyo_201802_daily_ave.rename(columns={1: 'Consumption'}, inplace=True)

fig, ax = plt.subplots(1,1)
ax.plot(juyo_201802_daily_ave)
ax.xaxis.set_major_locator(mdates.DayLocator([(i*2) for i in range(16)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
ax.set_title('Daily average Electric Power Consumption in 2018/02',fontsize=13)
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
fig.autofmt_xdate(rotation=90, ha='center')
plt.tight_layout()
plt.savefig("juyo_201802_daily_ave.png",dpi=300)
plt.show()

#%%
# z-score normalization function
def zscore(x):
    x_mean = x.mean(axis=0)
    x_std = np.std(x, axis=0)
    result = (x - x_mean)/x_std
    return result

#%%
# Make normalized juyo_201801_info
juyo_201801_info_normalized = zscore(juyo_201801_info)
print(juyo_201801_info_normalized)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_title('Electric Power Consumption in 2018/01 (Original)',fontsize=13)
ax2.set_title('Electric Power Consumption in 2018/01 (Normalized)',fontsize=13)
ax1.plot(juyo_201801_info)
ax2.plot(juyo_201801_info_normalized)
plt.tight_layout()
plt.show

#%%
# Temperature data set
tem_2016_info = pd.read_csv('temperature2016.csv')
tem_2016_info['datetime'] = pd.to_datetime(tem_2016_info['datetime'])
tem_2016_info = tem_2016_info.set_index('datetime') 
tem_2016_info = tem_2016_info['Temperature'].astype(float)
tem_2016_info = pd.DataFrame(tem_2016_info)

tem_2017_info = pd.read_csv('temperature2017.csv')
tem_2017_info['datetime'] = pd.to_datetime(tem_2017_info['datetime'])
tem_2017_info = tem_2017_info.set_index('datetime') 
tem_2017_info = tem_2017_info['Temperature'].astype(float)
tem_2017_info = pd.DataFrame(tem_2017_info)

tem_2018_info = pd.read_csv('temperature2018.csv')
tem_2018_info['datetime'] = pd.to_datetime(tem_2018_info['datetime'])
tem_2018_info = tem_2018_info.set_index('datetime') 
tem_2018_info = tem_2018_info['Temperature'].astype(float)
tem_2018_info = pd.DataFrame(tem_2018_info)

tem_info = pd.concat([tem_2016_info, tem_2017_info, tem_2018_info])

tem_info.to_pickle('tem_info.pkl')
tem_info.to_csv('tem_info.csv')

#%%
tem_info = tem_info.set_index([tem_info.index,
                               tem_info.index.year, 
                               tem_info.index.month, 
                               tem_info.index.weekday, 
                               tem_info.index.day, 
                               tem_info.index.hour])
tem_info.index.names = ['datetime','year', 'month', 'weekday','day', 'hour']

#%%
# concat all data

all_data = pd.concat([juyo_info, tem_info], axis=1)
#%%
# make daily average all_data in 2018/01

all_data_201801 = all_data.loc[pd.IndexSlice[:, 2018, 1],:]

#%%
tem_201801_daily_ave = []
for i in range(1, int(len(all_data_201801['Temperature'])/24+1)):
    xday_temp = all_data_201801.loc[pd.IndexSlice[:, 2018, 1, :, i], :]['Temperature']
    xday_temp.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=True)
    xday_ave = xday_temp.mean()
    
    tem_201801_daily_ave.append(xday_ave)

start_date = '2018-01-01'
end_date = '2018-01-31'
day_201801 = pd.date_range(start_date, end_date, freq='D')

tem_201801_daily_ave = pd.Series(tem_201801_daily_ave)
day_201801 = pd.Series(day_201801)
tem_201801_daily_ave = pd.concat([day_201801, tem_201801_daily_ave], axis=1)
tem_201801_daily_ave = tem_201801_daily_ave.set_index(0)
tem_201801_daily_ave.rename(columns={1: 'Temperature'}, inplace=True)

#%%
# Make normalized juyo_201801_daily_ave
# Make normalized tem_201801_daily_ave
juyo_201801_daily_ave_normalized = zscore(juyo_201801_daily_ave)
tem_201801_daily_ave_normalized = zscore(tem_201801_daily_ave)

#%%
# plot daily average juyo_info in 2018/01
# plot daily average tem_info in 2018/01

fig, ax = plt.subplots(1,1)
ax.plot(juyo_201801_daily_ave_normalized, label='Consumption')
ax.plot(tem_201801_daily_ave_normalized, label='Temperature')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*2) for i in range(16)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
ax.set_title('Comparison Daily average Consumption and Temperature \n in 2018/01',fontsize=13)
ax.set_ylabel('Normalized',fontsize=12)
fig.autofmt_xdate(rotation=90, ha='center')
plt.tight_layout()
plt.legend()
plt.savefig("juyo_201801_daily_ave.png",dpi=300)
plt.show()

#%%
# Normalized 2018/1 data

all_data_201801_normalized = zscore(all_data_201801)

#%%
# Set time length
# Make LSTM inputs data function
time_length = 24

def make_dataset(data):
    
    inputs_data = []

    for i in range(len(data)-time_length):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    inputs_target = data['Consumption'][time_length:]

    inputs_data_np = [np.array(inputs_data) for inputs_data in inputs_data]
    inputs_data_np = np.array(inputs_data_np)
    
    inputs_target_np = np.array(inputs_target).reshape(len(inputs_data), 1)

    return inputs_data_np, inputs_target_np

#%%
# LSTM_inputs_data -> training data, LSTM_inputs_target
LSTM_inputs_data_201801, LSTM_inputs_target_201801= make_dataset(all_data_201801_normalized)

#%%
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

#%%
length_of_sequence = LSTM_inputs_data_201801.shape[1]
in_neurons = 2
out_neurons = 1
n_hidden = 700

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_neurons), return_sequences=False))
model.add(Dense(out_neurons))
model.add(Activation('linear'))
optimizer = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

#%%
if os.path.isfile('LSTM_201801_weights.h5'):
    model.load_weights('LSTM_201801_weights.h5')
else:
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
    LSTM_201801_history = model.fit(LSTM_inputs_data_201801, LSTM_inputs_target_201801,
                                    batch_size=100,
                                    epochs=20,
                                    validation_split=0.1,
                                    callbacks=[early_stopping])
    model.save_weights('LSTM_201801_weights.h5')

    # plot cost function
    fig, ax = plt.subplots(1,1)
    ax.plot(LSTM_201801_history.epoch, LSTM_201801_history.history['loss'])
    ax.set_title('TrainingError')
    ax.set_ylabel('Mean Squared Error (MSE)',fontsize=12)
    ax.set_xlabel('# Epochs',fontsize=12)
    plt.tight_layout()
    plt.savefig("TrainingError.png",dpi=300)
    plt.show()
    
#%%
# predicted is power consumption volatility
# RMSE Toshiba 83.49[10000kW] https://www.toshiba.co.jp/about/press/2017_11/pr_j0801.htm
# plot training set and predict set

from sklearn.metrics import mean_squared_error

predicted_201801 = model.predict(LSTM_inputs_data_201801)
predicted_201801_pd = pd.Series(predicted_201801[:,0])
predicted_201801_pd.index = all_data_201801_normalized[time_length:].index

MSE_201801 = mean_squared_error(all_data_201801_normalized[time_length:]['Consumption'], predicted_201801_pd)
print(MSE_201801)

re_all_data_201801_normalized = all_data_201801_normalized.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)
re_predicted_201801_pd = predicted_201801_pd.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)

fig, ax = plt.subplots(1,1)
ax.plot(re_all_data_201801_normalized['Consumption'], label='Actual')
ax.plot(re_predicted_201801_pd, label='Predicted')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*7+1) for i in range(5)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Electric Power Consumption (Normalized)',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption in 2018/01',fontsize=13)
ax.annotate('MSE: %.3f'%MSE_201801, 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
plt.legend()
plt.savefig("juyo_201801_predicted.png",dpi=300)
plt.show

#%%
# normalized -> original scale
# zscore = (x-mean)/std
# x = std*zsocre + mean

juyo_201801_mean = all_data_201801['Consumption'].mean(axis=0)
juyo_201801_std = np.std(all_data_201801['Consumption'], axis=0)
predicted_201801_original_scale = juyo_201801_std * predicted_201801 + juyo_201801_mean

predicted_201801_original_scale_pd = pd.Series(predicted_201801_original_scale[:,0])
predicted_201801_original_scale_pd.index = all_data_201801[time_length:].index

MSE_201801_original_scale = mean_squared_error(all_data_201801[time_length:]['Consumption'], predicted_201801_original_scale_pd)
print(MSE_201801_original_scale)

re_all_data_201801 = all_data_201801.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)
re_predicted_201801_original_scale_pd = predicted_201801_original_scale_pd.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)

fig, ax = plt.subplots(1,1)
ax.plot(re_all_data_201801['Consumption'], label='Actual')
ax.plot(re_predicted_201801_original_scale_pd, label='Predicted')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*7+1) for i in range(5)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption in 2018/01',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201801_original_scale), 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
plt.legend()
plt.savefig("juyo_201801_predicted_original_scale.png",dpi=300)
plt.show

#%%
# Predict 2018/02

# Make normalized all_data_201802
all_data_201802 = all_data.loc[pd.IndexSlice[:, 2018, 2],:]
all_data_201802_normalized = zscore(all_data_201802)

re_all_data_201802 = all_data_201802.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)
re_all_data_201802_normalized = all_data_201802_normalized.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_title('Electric Power Consumption in 2018/02 (Original)',fontsize=13)
ax2.set_title('Electric Power Consumption in 2018/02 (Normalized)',fontsize=13)
ax1.plot(re_all_data_201802['Consumption'])
ax2.plot(re_all_data_201802_normalized['Consumption'])
plt.tight_layout()
plt.show

#%%
# Make inputs data for model.predict()
LSTM_inputs_data_201802, LSTM_inputs_target_201802 = make_dataset(all_data_201802_normalized)

# Predicte 2018/02 original scale uses weights is training set 2018/01
predicted_201802 = model.predict(LSTM_inputs_data_201802)
predicted_201802_pd = pd.Series(predicted_201802[:,0])
predicted_201802_pd.index = all_data_201802_normalized[time_length:].index

MSE_201802 = mean_squared_error(all_data_201802_normalized[time_length:]['Consumption'], predicted_201802_pd)
print(MSE_201802)

re_all_data_201802_normalized = all_data_201802_normalized.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)
re_predicted_201802_pd = predicted_201802_pd.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)


fig, ax = plt.subplots(1,1)
ax.plot(re_all_data_201802_normalized['Consumption'], label='Actual')
ax.plot(re_predicted_201802_pd, label='Predicted')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*7+1) for i in range(5)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Electric Power Consumption (Normalized)',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption in 2018/02',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201802), 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
plt.legend()
plt.savefig("juyo_201802_predicted.png",dpi=300)
plt.show
#%%
# normalized -> original scale
# Using last year data
all_data_201702 = all_data.loc[pd.IndexSlice[:, 2017, 2],:]
juyo_201702_mean = all_data_201702['Consumption'].mean(axis=0)
juyo_201702_std = np.std(all_data_201702['Consumption'], axis=0)
predicted_201802_original_scale = juyo_201702_std * predicted_201802 + juyo_201702_mean

predicted_201802_original_scale_pd = pd.Series(predicted_201802_original_scale[:,0])
predicted_201802_original_scale_pd.index = all_data_201802[time_length:].index

MSE_201802_original_scale = mean_squared_error(all_data_201802[time_length:]['Consumption'], predicted_201802_original_scale_pd)
print(MSE_201802_original_scale)

re_all_data_201802 = all_data_201802.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)
re_predicted_201802_original_scale_pd = predicted_201802_original_scale_pd.reset_index(level=['year','month','weekday','day','hour'], drop=True, inplace=False)

fig, ax = plt.subplots(1,1)
ax.plot(re_all_data_201802['Consumption'], label='Actual')
ax.plot(re_predicted_201802_original_scale_pd, label='Predicted')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*7+1) for i in range(5)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption in 2018/02',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201802_original_scale), 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
plt.legend()
plt.savefig("juyo_201802_predicted_original_scale.png",dpi=300)
plt.show

#%%
# future prediction

# start prediction data
# 1/31 because time_length=24
future_test = LSTM_inputs_data_201801[719]
# Variable saved future prediction data
future_result = np.empty((0))

test_data = future_test
for i in range(day_201802.shape[0]*24):
    test_data = np.reshape(future_test, (1, time_length, 2))
    batch_predict = model.predict(test_data)
    
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)

    future_result = np.append(future_result, batch_predict)

#%%
# plot future result
future_result_pd = pd.Series(future_result)

start_date = '2018-02-01 00:00:00'
end_date = '2018-02-28 23:00:00'
future_result_date = pd.date_range(start_date, end_date, freq='H')

future_result_pd.index = future_result_date

MSE_future_result = mean_squared_error(all_data_201802_normalized['Consumption'], future_result_pd)
print(MSE_201802)

fig, ax = plt.subplots(1,1)
ax.plot(re_all_data_201802_normalized['Consumption'], label='Actual')
ax.plot(future_result_pd, label='Predicted')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*7+1) for i in range(5)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Electric Power Consumption (Normalized)',fontsize=12)
ax.set_title('Future predict daily Electric Power Consumption in 2018/02',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_future_result), 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
plt.legend()
plt.savefig("future_predict_201802_normalized.png",dpi=300)
plt.show
#%%
# normalize -> original scale
future_result_original_scale = juyo_201702_std * future_result + juyo_201702_mean
future_result_original_scale_pd = pd.Series(future_result_original_scale)

future_result_original_scale_pd.index = future_result_date

#%%
# future MSE original scale
MSE_future_result_original_scale = mean_squared_error(all_data_201802['Consumption'], future_result_original_scale_pd)
print(MSE_future_result_original_scale)

#%%
# plot future result original scale
fig, ax = plt.subplots(1,1)
ax.plot(re_all_data_201802['Consumption'], label='Actual')
ax.plot(future_result_original_scale_pd, label='Future predict')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*7+1) for i in range(5)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
ax.set_title('Future predict daily Electric Power Consumption in 2018/02',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_future_result_original_scale), 
             xy=(0.75, 0.02),  xycoords='axes fraction',
            xytext=(0.75, 0.02), textcoords='axes fraction')
plt.legend()
plt.savefig("future_predict_201802_original_scale.png",dpi=300)
plt.show

#%%
# make daily average future predict in 2018/02
# plot daily average future predict in 2018/02
future_result_original_scale_pd_daily_ave = []
for i in range(1, int(len(future_result_original_scale_pd)/24 +1)):
    xday_temp = future_result_original_scale_pd[24*i:24*(i+1)]
    xday_ave = xday_temp.mean()
    
    future_result_original_scale_pd_daily_ave.append(xday_ave)

start_date = '2018-02-01'
end_date = '2018-02-28'
day_201802 = pd.date_range(start_date, end_date, freq='D')

future_result_original_scale_pd_daily_ave = pd.Series(future_result_original_scale_pd_daily_ave)
day_201802 = pd.Series(day_201802)
future_result_original_scale_pd_daily_ave = pd.concat([day_201802, future_result_original_scale_pd_daily_ave], axis=1)
future_result_original_scale_pd_daily_ave = future_result_original_scale_pd_daily_ave.set_index(0)
future_result_original_scale_pd_daily_ave.rename(columns={1: 'Consumption'}, inplace=True)

fig, ax = plt.subplots(1,1)
ax.plot(juyo_201802_daily_ave, label='Actual')
ax.plot(future_result_original_scale_pd_daily_ave, label='Future predict')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*2) for i in range(16)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
ax.set_title('Daily average Electric Power Consumption in 2018/02',fontsize=13)
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
fig.autofmt_xdate(rotation=90, ha='center')
plt.tight_layout()
plt.legend()
plt.savefig("future_predict_daily_average_201802.png",dpi=300)
plt.show()