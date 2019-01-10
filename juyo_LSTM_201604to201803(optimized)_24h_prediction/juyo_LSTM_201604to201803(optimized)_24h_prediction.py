import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#%%
juyo_info = pd.read_pickle('juyo_info.pkl')
tem_info = pd.read_pickle('tem_info.pkl')
re_im_info = pd.read_pickle('re_im_info.pkl')
re_spot_info = pd.read_pickle('re_spot_info.pkl')

#%%
# drop EreaPrice
re_spot_info.drop(['EreaPriceHokkaido(JPY/kWh)',
                   'EreaPriceTohoku(JPY/kWh)',
                   'EreaPriceChubu(JPY/kWh)',
                   'EreaPriceHokuriku(JPY/kWh)',
                   'EreaPriceKansai(JPY/kWh)',
                   'EreaPriceChugoku(JPY/kWh)',
                   'EreaPriceShikoku(JPY/kWh)',
                   'EreaPriceKyusyu(JPY/kWh)'], axis=1, inplace=True)

#%%
# concat all data

all_data = pd.concat([juyo_info, tem_info, re_im_info, re_spot_info], axis=1, join_axes=[juyo_info.index])
all_data = all_data.set_index([all_data.index,
                               all_data.index.year, 
                               all_data.index.month, 
                               all_data.index.weekday, 
                               all_data.index.day, 
                               all_data.index.hour])
all_data.index.names = ['datetime','year', 'month', 'weekday','day', 'hour']

#%%
# z-score normalization function
def zscore(training_data, data):
    training_data_mean = np.mean(training_data, axis=0)
    training_data_std = np.std(training_data, axis=0)
    normalized_data = (data - training_data_mean) / training_data_std
    return normalized_data, training_data_mean, training_data_std

#%%
# z-score normalization -> original scale function
def original_scale(predicted_data, training_data_mean, training_data_std):
    original_scale_predicted_data = training_data_std * predicted_data + training_data_mean
    return original_scale_predicted_data

#%%
# Set time length
# Make LSTM inputs data function
    
time_length = 24

def make_dataset(data):
    
    inputs_data = []
    
    for i in range(len(data)-time_length):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    
    inputs_target = np.zeros(shape=(len(data)-time_length, 2))
    for i in range(len(data)-time_length):
        if i < len(data)-time_length-1:
            for j in range(2):
                inputs_target[i, j] = data['Consumption'][time_length + i + j]
        else:
            inputs_target[i, 0] = data['Consumption'][time_length + i]

    inputs_data_np = [np.array(inputs_data) for inputs_data in inputs_data]
    inputs_data_np = np.array(inputs_data_np)
    
    inputs_target_np = np.array(inputs_target)

    return inputs_data_np, inputs_target_np

#%%
# make all_data in 2016/04 ~ 2018/03 and Normalized

all_data_2016 = all_data.loc[pd.IndexSlice[:, 2016, :],:]
all_data_2017 = all_data.loc[pd.IndexSlice[:, 2017, :],:]
all_data_201801to201803 = all_data.loc[pd.IndexSlice[:, 2018, 1:3],:]
all_data_201604to201803 = pd.concat([all_data_2016, all_data_2017, all_data_201801to201803])
all_data_201604to201803.reset_index(inplace=True)
all_data_201604to201803.set_index('datetime', inplace=True) 
all_data_201604to201803.drop('year', axis=1, inplace=True)

all_data_201604to201803.to_csv('all_data_201604to201803.csv')

all_data_201604to201803_normalized, all_data_201604to201803_mean, all_data_201604to201803_std = zscore(all_data_201604to201803, all_data_201604to201803)

LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803 = make_dataset(all_data_201604to201803_normalized)

#%%
# make all_data_201804 and Normalized

all_data_201804 = all_data.loc[pd.IndexSlice[:, 2018, 4],:]
all_data_201804.reset_index(inplace=True)
all_data_201804.set_index('datetime', inplace=True) 
all_data_201804.drop('year', axis=1, inplace=True)
all_data_201804_normalized, all_data_201804_mean, all_data_201804_std = zscore(all_data_201604to201803, all_data_201804)

LSTM_inputs_data_201804, LSTM_inputs_target_201804 = make_dataset(all_data_201804_normalized)


#%%
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

np.random.seed(123)

#%%
in_dim = LSTM_inputs_data_201604to201803.shape[2]
hidden_size = 858
out_dim = 1

model = Sequential()
model.add(CuDNNLSTM(hidden_size, return_sequences=True,
               batch_input_shape=(None, time_length, in_dim)))
model.add(CuDNNLSTM(hidden_size, return_sequences=False,
               batch_input_shape=(None, time_length, in_dim)))
model.add(Dense(out_dim, activation='linear'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
plot_model(model, to_file='model_201604to201803.png', show_shapes=True)

#%%
y_n = input("Use saved weight? [y/n] : ")
while True:
    if y_n == 'y':
        model.load_weights('best_model_checkpint.h5')
        break
    elif y_n == 'n':
        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
        model_checkpoint = ModelCheckpoint(filepath='best_model_checkpint.h5', monitor='val_loss', save_best_only=True)
        tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
        LSTM_201604to201803_history = model.fit(LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803,
                                batch_size=175,
                                epochs=5,
                                validation_split=0.1,
                                shuffle=False,
                                callbacks=[model_checkpoint,
                                           tensor_board])
        model.save_weights('LSTM_201604to201803_weights.h5')
        break
    else:
        y_n = input("Wrong input caracter. Use saved weight? [y/n] : ")

#%%
# plot cost function
base_line = mean_squared_error(LSTM_inputs_data_201604to201803[:,time_length-1,4], LSTM_inputs_target_201604to201803[:,0])
print('base line : %.5f'%base_line)

#%%
fig, ax = plt.subplots(1,1)
ax.plot(LSTM_201604to201803_history.epoch, LSTM_201604to201803_history.history['loss'], label='training loss')
ax.plot(LSTM_201604to201803_history.epoch, LSTM_201604to201803_history.history['val_loss'], label='validation loss')
ax.hlines(base_line, 0, len(LSTM_201604to201803_history.epoch), colors='r', linewidth=0.8, label='base line')
ax.annotate('base line: %.5f'%base_line, 
             xy=(0.72, 0.7),  xycoords='axes fraction',
            xytext=(0.72, 0.7), textcoords='axes fraction')
ax.set_title('model loss')
ax.set_ylabel('Mean Squared Error (MSE)',fontsize=12)
ax.set_xlabel('Epochs',fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("model_loss_201604to201803.png",dpi=300)
plt.show()

#%%
# model evaluate
# mean absokute percentage error
    
def mean_absolute_percentage_error(data, data_pred):
    MAPE = 100 * np.mean(np.abs((data - data_pred)/data_pred))
    return MAPE

#%%
# predicted is power consumption volatility
# RMSE Toshiba 83.49[10000kW] https://www.toshiba.co.jp/about/press/2017_11/pr_j0801.htm
# plot training set and predict set

predicted_201604to201803 = model.predict(LSTM_inputs_data_201604to201803)
predicted_201604to201803_pd = pd.Series(predicted_201604to201803[:,0])
predicted_201604to201803_pd.index = all_data_201604to201803_normalized[time_length:].index

MSE_201604to201803_1h = mean_squared_error(all_data_201604to201803_normalized[time_length:]['Consumption'], predicted_201604to201803_pd)

moving_average_length = np.ones(168)/168.0
all_data_201604to201803_normalized_moving_average = np.convolve(all_data_201604to201803_normalized['Consumption'], moving_average_length, mode='same')
all_data_201604to201803_normalized_moving_average = pd.DataFrame(data=all_data_201604to201803_normalized_moving_average, index=all_data_201604to201803.index)
predicted_201604to201803_pd_moving_average = np.convolve(predicted_201604to201803_pd, moving_average_length, mode='same')
predicted_201604to201803_pd_moving_average = pd.DataFrame(data=predicted_201604to201803_pd_moving_average, index=predicted_201604to201803_pd.index)

fig, ax = plt.subplots(1,1)
ax.plot(all_data_201604to201803_normalized['Consumption'], label='Actual', linewidth=0.8)
ax.plot(predicted_201604to201803_pd, label='Predicted', linewidth=0.8)
ax.plot(all_data_201604to201803_normalized_moving_average, label='Actual(Moving Average)')
ax.plot(predicted_201604to201803_pd_moving_average, label='Predicted(Moving Average)')
ax.xaxis.set_major_locator(mdates.MonthLocator([(i*2) for i in range(1,12)]))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
ax.set_ylabel('Electric Power Consumption (Normalized)',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption \n in 2016/04 ~ 2018/03',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201604to201803), 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
fig.autofmt_xdate(rotation=45, ha='center')
plt.legend()
plt.savefig("juyo_201604to201803_predicted.png",dpi=300)
plt.show

#%%
# normalized -> original scale
# zscore = (x-mean)/std
# x = std*zsocre + mean

predicted_201604to201803_original_scale =  original_scale(predicted_201604to201803, all_data_201604to201803_mean['Consumption'], all_data_201604to201803_std['Consumption'])
predicted_201604to201803_original_scale_pd = pd.Series(predicted_201604to201803_original_scale[:,0])
predicted_201604to201803_original_scale_pd.index = all_data_201604to201803[time_length:].index

MSE_201604to201803_original_scale = mean_squared_error(all_data_201604to201803[time_length:]['Consumption'], predicted_201604to201803_original_scale_pd)
MAPE_201604to201803_original_scale = mean_absolute_percentage_error(all_data_201604to201803[time_length:]['Consumption'], predicted_201604to201803_original_scale_pd)

moving_average_length = np.ones(168)/168.0
all_data_201604to201803_moving_average = np.convolve(all_data_201604to201803['Consumption'], moving_average_length, mode='same')
all_data_201604to201803_moving_average = pd.DataFrame(data=all_data_201604to201803_moving_average, index=all_data_201604to201803.index)
predicted_201604to201803_original_scale_pd_moving_average = np.convolve(predicted_201604to201803_original_scale_pd, moving_average_length, mode='same')
predicted_201604to201803_original_scale_pd_moving_average = pd.DataFrame(data=predicted_201604to201803_original_scale_pd_moving_average, index=predicted_201604to201803_original_scale_pd.index)

fig, ax = plt.subplots(1,1)
ax.plot(all_data_201604to201803['Consumption'], label='Actual', linewidth=0.8)
ax.plot(predicted_201604to201803_original_scale_pd, label='Predicted', linewidth=0.8)
ax.plot(all_data_201604to201803_moving_average, label='Actual(Moving Average)')
ax.plot(predicted_201604to201803_original_scale_pd_moving_average, label='Predicted(Moving Average)')
ax.xaxis.set_major_locator(mdates.MonthLocator([(i*2) for i in range(1,12)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption \n in 2016/04 ~ 2018/03',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201604to201803_original_scale), 
             xy=(0.78, 0.02),  xycoords='axes fraction',
            xytext=(0.78, 0.02), textcoords='axes fraction')
ax.annotate('MAPE: %.3f%%'%MAPE_201604to201803_original_scale, 
             xy=(0.78, 0.07),  xycoords='axes fraction',
            xytext=(0.78, 0.07), textcoords='axes fraction')
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
fig.autofmt_xdate(rotation=45, ha='center')
plt.legend()
plt.savefig("juyo_201604to201803_predicted_original_scale.png",dpi=300)
plt.show

#%%
# Predicte 2018/02 uses weights is training set 2016/04 ~ 2018/03

predicted_201804 = model.predict(LSTM_inputs_data_201804)
predicted_201804_pd = pd.Series(predicted_201804[:,0])
predicted_201804_pd.index = all_data_201804_normalized[time_length:].index

MSE_201804 = mean_squared_error(all_data_201804_normalized[time_length:]['Consumption'], predicted_201804_pd)

moving_average_length = np.ones(24)/24.0
all_data_201804_normalized_moving_average = np.convolve(all_data_201804_normalized['Consumption'], moving_average_length, mode='same')
all_data_201804_normalized_moving_average = pd.DataFrame(data=all_data_201804_normalized_moving_average, index=all_data_201804.index)
predicted_201804_pd_moving_average = np.convolve(predicted_201804_pd, moving_average_length, mode='same')
predicted_201804_pd_moving_average = pd.DataFrame(data=predicted_201804_pd_moving_average, index=predicted_201804_pd.index)

fig, ax = plt.subplots(1,1)
ax.plot(all_data_201804_normalized['Consumption'], label='Actual', linewidth=0.8)
ax.plot(predicted_201804_pd, label='Predicted', linewidth=0.8)
ax.plot(all_data_201804_normalized_moving_average, label='Actual(Moving Average)')
ax.plot(predicted_201804_pd_moving_average, label='Predicted(Moving Average)')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*3+1) for i in range(10)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.set_ylabel('Electric Power Consumption (Normalized)',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption in 2018/04',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201804), 
             xy=(0.76, 0.02),  xycoords='axes fraction',
            xytext=(0.76, 0.02), textcoords='axes fraction')
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
fig.autofmt_xdate(rotation=45, ha='center')
plt.legend()
plt.savefig("juyo_201804_predicted.png",dpi=300)
plt.show

#%%
# normalized -> original scale
# Using last year data

predicted_201804_original_scale =  original_scale(predicted_201804, all_data_201604to201803_mean['Consumption'], all_data_201604to201803_std['Consumption'])

predicted_201804_original_scale_pd = pd.Series(predicted_201804_original_scale[:,0])
predicted_201804_original_scale_pd.index = all_data_201804[time_length:].index

MSE_201804_original_scale = mean_squared_error(all_data_201804[time_length:]['Consumption'], predicted_201804_original_scale_pd)
MAPE_201804_original_scale = mean_absolute_percentage_error(all_data_201804[time_length:]['Consumption'], predicted_201804_original_scale_pd)

moving_average_length = np.ones(24)/24.0
all_data_201804_moving_average = np.convolve(all_data_201804['Consumption'], moving_average_length, mode='same')
all_data_201804_moving_average = pd.DataFrame(data=all_data_201804_moving_average, index=all_data_201804.index)
predicted_201804_original_scale_pd_moving_average = np.convolve(predicted_201804_original_scale_pd, moving_average_length, mode='same')
predicted_201804_original_scale_pd_moving_average = pd.DataFrame(data=predicted_201804_original_scale_pd_moving_average, index=predicted_201804_original_scale_pd.index)

fig, ax = plt.subplots(1,1)
ax.plot(all_data_201804['Consumption'], label='Actual', linewidth=0.8)
ax.plot(predicted_201804_original_scale_pd, label='Predicted', linewidth=0.8)
ax.plot(all_data_201804_moving_average, label='Actual(Moving Average)')
ax.plot(predicted_201804_original_scale_pd_moving_average, label='Predicted(Moving Average)')
ax.xaxis.set_major_locator(mdates.DayLocator([(i*3+1) for i in range(10)]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
ax.set_title('Predict daily Electric Power Consumption in 2018/04',fontsize=13)
ax.annotate('RMSE: %.3f'%np.sqrt(MSE_201804_original_scale), 
             xy=(0.76, 0.02),  xycoords='axes fraction',
            xytext=(0.76, 0.02), textcoords='axes fraction')
ax.annotate('MAPE: %.3f%%'%MAPE_201804_original_scale, 
             xy=(0.76, 0.07),  xycoords='axes fraction',
            xytext=(0.76, 0.07), textcoords='axes fraction')
ax.grid(which='major',color='gray',linestyle='--')
ax.grid(which='minor',color='gray',linestyle='--')
fig.autofmt_xdate(rotation=45, ha='center')
plt.legend()
plt.savefig("juyo_201804_predicted_original_scale_last_year.png",dpi=300)
plt.show