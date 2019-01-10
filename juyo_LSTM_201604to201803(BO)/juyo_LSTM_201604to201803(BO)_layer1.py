import pandas as pd
import numpy as np

#%%
juyo_info = pd.read_pickle('juyo_info.pkl')
tem_info = pd.read_pickle('tem_info.pkl')
re_im_info = pd.read_pickle('re_im_info.pkl')
re_spot_info = pd.read_pickle('re_spot_info.pkl')

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
from keras.callbacks import EarlyStopping
from bayes_opt_custom import BayesianOptimization

np.random.seed(123)

#%%
def model_by_BayesianOptimization(hidden_size_1,                                                           
                                  batch_size):
    
    in_dim = LSTM_inputs_data_201604to201803.shape[2]
    hidden_size_1 = int(hidden_size_1)
    out_dim = 1


    model = Sequential()
    
    model.add(CuDNNLSTM(hidden_size_1, return_sequences=False,
                   batch_input_shape=(None, time_length, in_dim)))        
    activation = 'linear'
    model.add(Dense(out_dim, activation=activation))
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    LSTM_201604to201803_history = model.fit(LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803,
                                            batch_size=int(batch_size),
                                            epochs=25,
                                            validation_split=0.1,
                                            shuffle=False,
                                            callbacks=[early_stopping])
    
    return -LSTM_201604to201803_history.history['val_loss'][-1]

#%%
# BayesianOptimization
    
pbounds = {'hidden_size_1': (32, 1024),
           'batch_size': (32, 1024)}

optimizer = BayesianOptimization(f=model_by_BayesianOptimization, pbounds=pbounds)

optimizer.maximize(init_points=10, n_iter=200, acq='ei')

#%%
all_BO = optimizer.res['all']
#%%
max_BO = optimizer.res['max']

#%%
"""
import pickle

with open('all_BO.pickle', 'wb') as f:
    pickle.dump(all_BO, f)
"""

all_BO_values = np.zeros(189)
all_BO_hidden = np.zeros(189)
all_BO_batch = np.zeros(189)
for i in range(189):
    all_BO_values[i] = all_BO['values'][i]
    all_BO_hidden[i] = all_BO['params'][i]['hidden_size_1']
    all_BO_batch[i] = all_BO['params'][i]['batch_size']
    
all_BO_np = np.vstack((all_BO_values, all_BO_hidden, all_BO_batch)).T

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = all_BO_np[:,1]
y = all_BO_np[:,2]
z = - all_BO_np[:,0]

ax.scatter(x, y, z)
plt.show()