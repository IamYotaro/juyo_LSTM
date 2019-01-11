import pandas as pd
import numpy as np
import os

PATH = 'E:\AnacondaProjects\juyo_LSTM\juyo_LSTM_data'
os.chdir(PATH)

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
# Temperature data set
tem_info = pd.read_csv('temperature.csv')
tem_info['datetime'] = pd.to_datetime(tem_info['datetime'])
tem_info = tem_info.set_index('datetime') 
tem_info = tem_info['Temperature'].astype(float)
tem_info = pd.DataFrame(tem_info)

tem_info.to_pickle('tem_info.pkl')
tem_info.to_csv('tem_info.csv')

#%%
# Kosuiryo data set
kos_info = pd.read_csv('kosuiryo.csv')
kos_info['datetime'] = pd.to_datetime(kos_info['datetime'])
kos_info = kos_info.set_index('datetime') 
kos_info = kos_info['Kosuiryo(mm)'].astype(float)
kos_info = pd.DataFrame(kos_info)
#
kos_info.to_pickle('kos_info.pkl')
kos_info.to_csv('kos_info.csv')

#%%
# Nissha data set
nis_info = pd.read_csv('nisha.csv')
nis_info['datetime'] = pd.to_datetime(nis_info['datetime'])
nis_info = nis_info.set_index('datetime') 
nis_info = nis_info['Nisha(MJ/m2)'].astype(float)
nis_info = pd.DataFrame(nis_info)

nis_info.to_pickle('nis_info.pkl')
nis_info.to_csv('nis_info.csv')

#%%
# Nissha data set
sit_info = pd.read_csv('sitsudo.csv')
sit_info['datetime'] = pd.to_datetime(sit_info['datetime'])
sit_info = sit_info.set_index('datetime') 
sit_info = sit_info['Sitsudo(%)'].astype(float)
sit_info = pd.DataFrame(sit_info)

sit_info.to_pickle('sit_info.pkl')
sit_info.to_csv('sit_info.csv')
#%%
# JEPX 1 hour ago market

im_info = pd.read_csv('im_trade_summary.csv')
im_info.fillna(0, inplace=True)
start_date_market = '2016-04-01 00:00:00'
end_date_market = '2018-03-31 23:30:00'
day_market = pd.date_range(start_date_market, end_date_market, freq='0.5H')
day_market = pd.Series(day_market)
im_info.drop(['date',
              'TimeCode',
              'Open(JPY/kWh)',
              'High(JPY/kWh)',
              'Low(JPY/kWh)',
              'Average(JPY/kWh)',], axis=1, inplace=True)
im_info = pd.concat([day_market, im_info], axis=1)
im_info = im_info.set_index(0)

im_info.to_pickle('im_info.pkl')
im_info.to_csv('im_info.csv')

#%%
# per 30minuts -> per 1hour
# but convert numpy

def re_data(data):
    
    data = data.values
    re_data = []
    
    for i in range(int(len(data)/2)):
        unit_data = (data[i*2] + data[i*2+1]) / 2
        re_data = np.concatenate([re_data, unit_data], axis=0)
    
    re_data = np.reshape(re_data, (-1, data.shape[1]))
    
    return re_data

#%%
# numpy -> pandas
    
re_im_info = re_data(im_info)
re_im_info = pd.DataFrame(data=re_im_info, columns=im_info.columns)

start_re_date_market = '2016-04-01 00:00:00'
end_re_date_market = '2018-10-07 23:00:00'
re_day_market = pd.date_range(start_re_date_market, end_re_date_market, freq='H')
re_day_market = pd.Series(re_day_market)
re_im_info = pd.concat([re_day_market, re_im_info], axis=1)
re_im_info = re_im_info.set_index(0)

re_im_info.to_pickle('re_im_info.pkl')
re_im_info.to_csv('re_im_info.csv')

#%%
spot_info = pd.read_csv('spot.csv')
spot_info.fillna(0, inplace=True)
spot_info.drop(['date',
                'TimeCode',
                'ShortVolume(kWh)',
                'LongVolume(kWh)',
                'TotalTransactionVolume(spot)(MWh/h)',
                'SystemPrice(JPY/kWh)',
                'EreaPriceHokkaido(JPY/kWh)',
                'EreaPriceTohoku(JPY/kWh)',
                'EreaPriceChubu(JPY/kWh)',
                'EreaPriceHokuriku(JPY/kWh)',
                'EreaPriceKansai(JPY/kWh)',
                'EreaPriceChugoku(JPY/kWh)',
                'EreaPriceShikoku(JPY/kWh)',
                'EreaPriceKyusyu(JPY/kWh)'], axis=1, inplace=True)
spot_info = pd.concat([day_market, spot_info], axis=1)
spot_info = spot_info.set_index(0)

spot_info.to_pickle('spot_info.pkl')
spot_info.to_csv('spot_info.csv')

#%%
# numpy -> pandas
    
re_spot_info = re_data(spot_info)
re_spot_info = pd.DataFrame(data=re_spot_info, columns=spot_info.columns)

re_spot_info = pd.concat([re_day_market, re_spot_info], axis=1)
re_spot_info = re_spot_info.set_index(0)

re_spot_info.to_pickle('re_spot_info.pkl')
re_spot_info.to_csv('re_spot_info.csv')
    
#%%
# concat all data

all_data = pd.concat([juyo_info, tem_info, kos_info, nis_info, sit_info, re_im_info, re_spot_info], axis=1, join_axes=[juyo_info.index])
all_data.set_index([all_data.index,
                    all_data.index.year, 
                    all_data.index.month, 
                    all_data.index.weekday, 
                    all_data.index.hour], inplace=True)
all_data.index.names = ['datetime','year', 'month', 'weekday', 'hour']
all_data.reset_index(inplace=True)
all_data.set_index('datetime', inplace=True)
all_data.fillna(0, inplace=True)
all_data.to_csv('all_data.csv')