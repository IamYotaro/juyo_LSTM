import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from tqdm import tqdm
import pickle

#%%
PATH = 'E:/AnacondaProjects/juyo_LSTM/juyo_LSTM_feature_select'
os.chdir(PATH)
all_data_201604to201803 = pd.read_csv('all_data_201604to201803.csv')

#%%
all_data_201604to201803 = pd.read_csv('all_data_201604to201803.csv')
all_data_201604to201803.set_index('datetime', inplace=True)
all_data_201604to201803.drop('day', axis=1, inplace=True)

#%%
fixed_list = ['month','weekday','hour','Consumption','Temperature']
def DataCombination(data):
    data_fixed = data.loc[:,fixed_list]
    data_not_fixed = data.drop(fixed_list, axis=1)

    not_fixed_columns = data_not_fixed.columns
    not_fixed_column_list = []
    for i in range(len(not_fixed_columns)):
        not_fixed_column_list.append(not_fixed_columns[i])
    
    column_comb = []
    for i in range(data_not_fixed.shape[1]):
        temp_column_comb = list(itertools.combinations(not_fixed_column_list, i+1))
        column_comb.extend(temp_column_comb)
    
    return data_fixed, column_comb
    
#%%
# z-score normalization function
def zscore(training_data, data):
    training_data_mean = np.mean(training_data, axis=0)
    training_data_std = np.std(training_data, axis=0)
    normalized_data = (data - training_data_mean) / training_data_std
    return normalized_data, training_data_mean, training_data_std

#%%    
time_length = 24
n_pred = 1

def make_dataset(data):
    
    inputs_data = []
    
    for i in range(len(data)-time_length-n_pred+1):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    
    inputs_target = np.zeros(shape=(len(data)-time_length-n_pred+1, n_pred))
    for i in range(len(data)-time_length-n_pred+1):
        for j in range(n_pred):
            inputs_target[i, j] = data['Consumption'][time_length + i + j]

    inputs_data_np = [np.array(inputs_data) for inputs_data in inputs_data]
    inputs_data_np = np.array(inputs_data_np)
    
    inputs_target_np = np.array(inputs_target)

    return inputs_data_np, inputs_target_np

#%%
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM
from keras import optimizers
from keras.callbacks import EarlyStopping

np.random.seed(123)

#%%    
def LSTM_model(LSTM_inputs_data, LSTM_inputs_target):
    in_dim = LSTM_inputs_data.shape[2]
    hidden_size = 256
    out_dim = 1


    model = Sequential()
    
    model.add(CuDNNLSTM(hidden_size, return_sequences=False,
                        batch_input_shape=(None, time_length, in_dim)))        
    activation = 'linear'
    model.add(Dense(out_dim, activation=activation))
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    LSTM_history = model.fit(LSTM_inputs_data, LSTM_inputs_target,
                             batch_size=512,
                             epochs=25,
                             validation_split=0.1,
                             shuffle=False,
                             callbacks=[early_stopping],
                             verbose=1)
    
    return LSTM_history.history['val_loss'][-1]

#%%

all_data_201604to201803_fixed_feature, column_comb = DataCombination(all_data_201604to201803)    
'''
# fixed data only
all_data_201604to201803_comb = all_data_201604to201803_fixed_feature
all_data_201604to201803_comb_normalized, all_data_201604to201803_comb_mean, all_data_201604to201803_comb_std = zscore(all_data_201604to201803_comb, all_data_201604to201803_comb)
LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803 = make_dataset(all_data_201604to201803_comb_normalized)
    
column_comb_MSE_fixed_only = LSTM_model(LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803)
    
with open('column_comb_MSE_fixed_only.pickle', mode='wb') as f:
    pickle.dump(column_comb_MSE_fixed_only, f)

# fixed data and data combination
column_comb_MSE = np.zeros(len(column_comb))
for i in tqdm(range(4090, len(column_comb))):
    all_data_201604to201803_comb = pd.concat([all_data_201604to201803_fixed_feature,
                                              all_data_201604to201803.loc[:,column_comb[i]]], axis=1)
    all_data_201604to201803_comb_normalized, all_data_201604to201803_comb_mean, all_data_201604to201803_comb_std = zscore(all_data_201604to201803_comb, all_data_201604to201803_comb)
    LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803 = make_dataset(all_data_201604to201803_comb_normalized)
    
    column_comb_MSE[i] = LSTM_model(LSTM_inputs_data_201604to201803, LSTM_inputs_target_201604to201803)
    
    if i+1 >= 10 and (i+1) % 10 == 0:
        with open('column_comb_MSE.pickle', mode='wb') as f:
            pickle.dump(column_comb_MSE, f)
'''
#%%

def delete_zero(data):
    deleted_data = np.delete(data, np.where(data == 0))
    return deleted_data

column_comb_MSE_fixed_only = delete_zero(pd.read_pickle('column_comb_MSE_fixed_only.pickle')) 
column_comb_MSE_0_749 = delete_zero(pd.read_pickle('column_comb_MSE_0_749.pickle'))
column_comb_MSE_750_879 = delete_zero(pd.read_pickle('column_comb_MSE_750_879.pickle'))
column_comb_MSE_880_1249 = delete_zero(pd.read_pickle('column_comb_MSE_880_1249.pickle'))
column_comb_MSE_1250_1839 = delete_zero(pd.read_pickle('column_comb_MSE_1250_1839.pickle'))
column_comb_MSE_1840_2659 = delete_zero(pd.read_pickle('column_comb_MSE_1840_2659.pickle'))
column_comb_MSE_2660_3379 = delete_zero(pd.read_pickle('column_comb_MSE_2660_3379.pickle'))
column_comb_MSE_3380_4089 = delete_zero(pd.read_pickle('column_comb_MSE_3380_4089.pickle'))
column_comb_MSE_4090_4094 = delete_zero(pd.read_pickle('column_comb_MSE_4090_4094.pickle'))

column_comb_MSE = np.hstack((column_comb_MSE_fixed_only,
                             column_comb_MSE_0_749,
                             column_comb_MSE_750_879,
                             column_comb_MSE_880_1249,
                             column_comb_MSE_1250_1839,
                             column_comb_MSE_1840_2659,
                             column_comb_MSE_2660_3379,
                             column_comb_MSE_3380_4089,
                             column_comb_MSE_4090_4094))

#%%

fig, ax = plt.subplots(1,1)
ax.plot(column_comb_MSE, linewidth=1, zorder=1)
ax.hlines(column_comb_MSE_fixed_only, 0, len(column_comb_MSE)-1, colors='r', linewidth=0.8, zorder=2)
ax.annotate('base line: %.5f'%column_comb_MSE_fixed_only, 
             xy=(0.72, 0.02),  xycoords='axes fraction',
            xytext=(0.72, 0.02), textcoords='axes fraction')
ax.set_xlabel('Numbers',fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE)',fontsize=12)
ax.set_title('Feature Combination',fontsize=13)
plt.savefig("FeatureCombination.png",dpi=300)
plt.show

column_comb_MSE_min = column_comb_MSE[np.argmin(column_comb_MSE)]
column_comb_MSE_feature = column_comb[np.argmin(column_comb_MSE)-1]

#%%

column_comb_MSE_list = pd.DataFrame(np.vstack((np.array(column_comb), column_comb_MSE[1:]))).T
column_comb_MSE_list.set_index(0, inplace=True)
column_comb_MSE_list.rename(columns={1: 'MSE'}, inplace=True)
column_comb_MSE_list.sort_values(by='MSE', inplace=True)