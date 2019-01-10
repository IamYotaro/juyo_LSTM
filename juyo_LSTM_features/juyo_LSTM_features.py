import pandas as pd
import matplotlib.pyplot as plt

#%%
all_data_201604to201803 = pd.read_csv('all_data_201604to201803.csv')
all_data_201604to201803.set_index('datetime', inplace=True)

#%%
y_n = input("print Temperature-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Temperature']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Temperature[\u00B0C]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('Temperature.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Temperature-Consumption? [y/n] : ")        

#%%
y_n = input("print month-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['month']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Month')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('month.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print month-Consumption? [y/n] : ")
        
#%%
y_n = input("print weekday-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['weekday']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Weekday')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('weekday.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print weekday-Consumption? [y/n] : ")

#%%
y_n = input("print day-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['day']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Day')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('day.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print day-Consumption? [y/n] : ")

#%%
y_n = input("print hour-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['hour']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('hour.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print hour-Consumption? [y/n] : ")

#%%
y_n = input("print month-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['month']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Month')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('month.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print month-Consumption? [y/n] : ")
        
#%%
y_n = input("print Open-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Open(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Open [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('Open.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Open-Consumption? [y/n] : ")
        
#%%
y_n = input("print High-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['High(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('High [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('High.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print High-Consumption? [y/n] : ")

#%%
y_n = input("print Low-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Low(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Low [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('Low.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Low-Consumption? [y/n] : ")

#%%
y_n = input("print Close-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Close(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Close [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('Close.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Close-Consumption? [y/n] : ")

#%%
y_n = input("print Average-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Average(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Average [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('Average.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Average-Consumption? [y/n] : ")
        
#%%
y_n = input("print TTV(im)-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['TotalTransactionVolume(im)(MWh/h)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Total Transaction Volume (im) [MWh/h]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('TTV(im).png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print TTV(im)-Consumption? [y/n] : ")

#%%
y_n = input("print TransactionNumber-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['TransactionNumber']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Transaction Number')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('TransactionNumber.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print TransactionNumber-Consumption? [y/n] : ")
        
#%%
y_n = input("print ShortVolume-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['ShortVolume(kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Short Volume [kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('ShortVolume.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print ShortVolume-Consumption? [y/n] : ")

#%%
y_n = input("print LongVolume-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['LongVolume(kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('LongVolume [kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('LongVolume.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print LongVolume-Consumption? [y/n] : ")

#%%
y_n = input("print TTV(spot)-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['TotalTransactionVolume(spot)(MWh/h)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('TotalTransactionVolume (spot) [MWh/h]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('TTV(spot).png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print TTV(spot)-Consumption? [y/n] : ")
        
#%%
y_n = input("print SystemPrice-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['SystemPrice(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('System Price [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('SystemPrice.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print SystemPrice-Consumption? [y/n] : ")

#%%
y_n = input("print EreaPriceTokyo-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['EreaPriceTokyo(JPY/kWh)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Erea Price Tokyo [JPY/kWh]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('EreaPriceTokyo.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print EreaPriceTokyo-Consumption? [y/n] : ")
        
#%%
y_n = input("print Kosuiryo-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Kosuiryo(mm)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Rainfall Amount [mm]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('ReinfallAmount.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Kosuiryo-Consumption? [y/n] : ")

#%%
y_n = input("print Nisha-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Nisha(MJ/m2)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Solar Radiation [MJ/m2]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('SolarRadiation.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Nisha-Consumption? [y/n] : ")
        
#%%
y_n = input("print Sitsudo-Consumption? [y/n] : ")
while True:
    if y_n == 'y':
        fig, ax = plt.subplots(1,1)
        plt.scatter(all_data_201604to201803['Sitsudo(%)']['2017-04-01 00:00:00':],
                    all_data_201604to201803['Consumption']['2017-04-01 00:00:00':],
                    marker='.')
        ax.set_xlabel('Humidity [%]')
        ax.set_ylabel('Electric Power Consumption [10000kW]',fontsize=12)
        plt.savefig('Humidity.png',dpi=300)
        plt.show
        break
    elif y_n == 'n':
        break
    else:
        y_n = input("wrong input character. print Sitsudo-Consumption? [y/n] : ")