#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pytensor import tensor as pt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import betabinom
from pymc_marketing.mmm.transformers import geometric_adstock
from sklearn.preprocessing import MinMaxScaler
#from src.scalers import MaxAbsScaler


def saturate(x, a):
    return 1 - pt.exp(-a*x)


# Function for creating the dataset
def create_weekly_dataset():
    start_date = datetime(2023, 1, 2)  # monday first week 2023
    end_date = datetime(2025, 12, 29)  # last monday in 2025
    data = []
    kw_year_counter = 1
    current_year = 2023
    current_date = start_date

    distribution = 45.0
    target_distribution = 90.0

    np.random.seed(42)

    week_count = 0

    while current_date <= end_date:
        iso_year, iso_week, _ = current_date.isocalendar()
        kw_format = f"{iso_year}-{iso_week:02d}"

        if iso_year != current_year:
            current_year = iso_year
            kw_year_counter = 1

        if distribution < target_distribution:
            rand_value = np.random.random()
            if rand_value < 0.40:
                increase = 0
            elif rand_value < 0.70:
                increase = np.random.uniform(0.3, 0.5)
            elif rand_value < 0.9:
                increase = np.random.uniform(0.5, 0.9)
            else:
                increase = np.random.uniform(0.9, 1.2)

            distribution = min(distribution + increase, target_distribution)

        data.append({
            'date': current_date,
            'KW': kw_format,
            'KW_year': kw_year_counter,
            'distribution': round(distribution, 2)
        })

        current_date += timedelta(weeks=1)
        kw_year_counter += 1
        week_count += 1

    df = pd.DataFrame(data)
    return df

# Create dataset
df = create_weekly_dataset()
df['Marke'] = 'MMT'

all_vars=['distribution','season','channel_A','channel_B','channel_C']
media_vars=['channel_A','channel_B','channel_C']


# Output
print("Data record created:")
print(f"\nNumber of rows: {len(df)}")
print("\nFirst 10 lines:")
print(df.head(10))
print("\nLast 10 lines:")
print(df.tail(10))
print("\nStatistics for distribution:")
print(df['distribution'].describe())


#add weather data
wetter=pd.read_excel('wetter.xlsx')
df=df.merge(wetter[['KW','SONNENSCHEINDAUER']])
df=df.rename(columns={'SONNENSCHEINDAUER': 'season'})


#add media data - evenly split
total_budget=15000000
media_split=np.array([0.7,0.2,0.1])
campaign_weeks=np.append(np.array(range(8,16,1)),np.array(range(32,40,1)))

campaign_budget=total_budget/6
campaign_split=campaign_budget*media_split
campaign_dist=betabinom.pmf(np.arange(0, 8), 7, 2, 3)

df_media_even=df
channel_A=campaign_split[0]*campaign_dist
channel_B=campaign_split[1]*campaign_dist
channel_C=campaign_split[2]*campaign_dist

df_media_even['channel_A'] = 0
week_to_budget = {}
for i, week in enumerate(range(8, 16)):
    week_to_budget[week] = channel_A[i]
for i, week in enumerate(range(32, 40)):
    week_to_budget[week] = channel_A[i]
df_media_even['channel_A'] = df_media_even['KW_year'].map(week_to_budget).fillna(0).round()

df_media_even['channel_B'] = 0
week_to_budget = {}
for i, week in enumerate(range(8, 16)):
    week_to_budget[week] = channel_B[i]
for i, week in enumerate(range(32, 40)):
    week_to_budget[week] = channel_B[i]
df_media_even['channel_B'] = df_media_even['KW_year'].map(week_to_budget).fillna(0).round()

df_media_even['channel_C'] = 0
week_to_budget = {}
for i, week in enumerate(range(8, 16)):
    week_to_budget[week] = channel_C[i]
for i, week in enumerate(range(32, 40)):
    week_to_budget[week] = channel_C[i]
df_media_even['channel_C'] = df_media_even['KW_year'].map(week_to_budget).fillna(0).round()

#scaling media variables
scaler_geo_dict={}
for media_channel in media_vars:
    scaler = MinMaxScaler()
    df_media_even[f'{media_channel}_sc'] = scaler.fit_transform(df_media_even[[media_channel]]).flatten()
    scaler_geo_dict[media_channel] = scaler


#calc contributions
ads_A=0.75
ads_B=0.5
ads_C=0.2

sat_A=3
sat_B=2
sat_C=2

coefficients=[0.6,1.3,0.8]



df_media_even['channel_A_tf'] = saturate(geometric_adstock(df_media_even['channel_A_sc'], ads_A).eval(),sat_A).eval()
df_media_even['channel_B_tf'] = saturate(geometric_adstock(df_media_even['channel_B_sc'], ads_B).eval(),sat_B).eval()
df_media_even['channel_C_tf'] = saturate(geometric_adstock(df_media_even['channel_C_sc'], ads_C).eval(),sat_C).eval()

#plt.plot(df_media_even['channel_A_sc'],df_media_even['channel_A_tf'])
#plt.plot(df_media_even['channel_B_sc'],df_media_even['channel_B_tf'])
#plt.plot(df_media_even['channel_C_sc'],df_media_even['channel_C_tf'])

#calculate contribution for media
i=0
for media_channel in media_vars:
    scaler = scaler_geo_dict[media_channel]
    df_media_even[f'{media_channel}_contr'] = np.array(scaler.inverse_transform(df_media_even[[f'{media_channel}_tf']])*coefficients[i]).flatten().round()
    i=i+1

df_media_even['distribution_contr']=df_media_even['distribution']*5000
df_media_even['season_contr']=df_media_even['season']*1500

df_media_even['umsatz']=df_media_even[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_even=df_media_even[['date','KW','KW_year','Marke','umsatz','distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr','distribution','season','channel_A','channel_B','channel_C']]



#dynamic media data
campaign_budget = np.array([0.08,0.11,0.14,0.16,0.24,0.27])*total_budget
media_split_1=np.array([0.7,0.2,0.1])
media_split_2=np.array([0.6,0.25,0.15])
media_split_3=np.array([0.55,0.27,0.18])

df_media_dyn=df
channel_A1=campaign_budget[0]*media_split_1[0]*campaign_dist
channel_A2=campaign_budget[1]*media_split_1[0]*campaign_dist
channel_A3=campaign_budget[2]*media_split_2[0]*campaign_dist
channel_A4=campaign_budget[3]*media_split_2[0]*campaign_dist
channel_A5=campaign_budget[4]*media_split_3[0]*campaign_dist
channel_A6=campaign_budget[5]*media_split_3[0]*campaign_dist

channel_B1=campaign_budget[0]*media_split_1[1]*campaign_dist
channel_B2=campaign_budget[1]*media_split_1[1]*campaign_dist
channel_B3=campaign_budget[2]*media_split_2[1]*campaign_dist
channel_B4=campaign_budget[3]*media_split_2[1]*campaign_dist
channel_B5=campaign_budget[4]*media_split_3[1]*campaign_dist
channel_B6=campaign_budget[5]*media_split_3[1]*campaign_dist

channel_C1=campaign_budget[0]*media_split_1[2]*campaign_dist
channel_C2=campaign_budget[1]*media_split_1[2]*campaign_dist
channel_C3=campaign_budget[2]*media_split_2[2]*campaign_dist
channel_C4=campaign_budget[3]*media_split_2[2]*campaign_dist
channel_C5=campaign_budget[4]*media_split_3[2]*campaign_dist
channel_C6=campaign_budget[5]*media_split_3[2]*campaign_dist


df_media_dyn['channel_A'] = 0
df_media_dyn['channel_B'] = 0
df_media_dyn['channel_C'] = 0

df_media_dyn.loc[df_media_dyn['KW'].between('2023-08', '2023-15'), 'channel_A'] = channel_A1.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2023-32', '2023-39'), 'channel_A'] = channel_A2.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2024-08', '2024-15'), 'channel_A'] = channel_A3.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2024-32', '2024-39'), 'channel_A'] = channel_A4.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2025-08', '2025-15'), 'channel_A'] = channel_A5.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2025-32', '2025-39'), 'channel_A'] = channel_A6.astype(int)

df_media_dyn.loc[df_media_dyn['KW'].between('2023-08', '2023-15'), 'channel_B'] = channel_B1.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2023-32', '2023-39'), 'channel_B'] = channel_B2.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2024-08', '2024-15'), 'channel_B'] = channel_B3.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2024-32', '2024-39'), 'channel_B'] = channel_B4.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2025-08', '2025-15'), 'channel_B'] = channel_B5.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2025-32', '2025-39'), 'channel_B'] = channel_B6.astype(int)

df_media_dyn.loc[df_media_dyn['KW'].between('2023-08', '2023-15'), 'channel_C'] = channel_C1.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2023-32', '2023-39'), 'channel_C'] = channel_C2.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2024-08', '2024-15'), 'channel_C'] = channel_C3.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2024-32', '2024-39'), 'channel_C'] = channel_C4.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2025-08', '2025-15'), 'channel_C'] = channel_C5.astype(int)
df_media_dyn.loc[df_media_dyn['KW'].between('2025-32', '2025-39'), 'channel_C'] = channel_C6.astype(int)


#scaling media variables
scaler_geo_dict={}
for media_channel in media_vars:
    scaler = MinMaxScaler()
    df_media_dyn[f'{media_channel}_sc'] = scaler.fit_transform(df_media_dyn[[media_channel]]).flatten()
    scaler_geo_dict[media_channel] = scaler
    
df_media_dyn['channel_A_tf'] = saturate(geometric_adstock(df_media_dyn['channel_A_sc'], ads_A).eval(),sat_A).eval()
df_media_dyn['channel_B_tf'] = saturate(geometric_adstock(df_media_dyn['channel_B_sc'], ads_B).eval(),sat_B).eval()
df_media_dyn['channel_C_tf'] = saturate(geometric_adstock(df_media_dyn['channel_C_sc'], ads_C).eval(),sat_C).eval()
  
#calculate contribution for media dynamic
i=0
for media_channel in media_vars:
    scaler = scaler_geo_dict[media_channel]
    df_media_dyn[f'{media_channel}_contr'] = np.array(scaler.inverse_transform(df_media_dyn[[f'{media_channel}_tf']])*coefficients[i]).flatten().round()
    i=i+1

df_media_dyn['distribution_contr']=df_media_dyn['distribution']*5000
df_media_dyn['season_contr']=df_media_dyn['season']*1500

df_media_dyn['umsatz']=df_media_dyn[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_dyn=df_media_dyn[['date','KW','KW_year','Marke','umsatz','distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr','distribution','season','channel_A','channel_B','channel_C']]



#media with time shift
df_media_shift=df

df_media_shift['channel_A'] = 0
df_media_shift['channel_B'] = 0
df_media_shift['channel_C'] = 0

df_media_shift.loc[df_media_shift['KW'].between('2023-08', '2023-15'), 'channel_A'] = (channel_A1+(channel_A5/5)).astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2023-32', '2023-39'), 'channel_A'] = (channel_A2+(channel_A5/5)).astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2024-08', '2024-15'), 'channel_A'] = (channel_A3+(channel_A5/5)).astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2024-32', '2024-39'), 'channel_A'] = (channel_A4+(channel_A5/5)).astype(int)
#df_media_shift.loc[df_media_shift['KW'].between('2025-10', '2025-17'), 'channel_A'] = channel_A5.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2025-32', '2025-39'), 'channel_A'] = (channel_A6+(channel_A5/5)).astype(int)

df_media_shift.loc[df_media_shift['KW'].between('2023-10', '2023-17'), 'channel_B'] = channel_B1.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2023-32', '2023-39'), 'channel_B'] = channel_B2.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2024-10', '2024-17'), 'channel_B'] = (channel_B3+(channel_B4/3)).astype(int)
#df_media_shift.loc[df_media_shift['KW'].between('2024-36', '2024-43'), 'channel_B'] = channel_B4.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2025-08', '2025-15'), 'channel_B'] = (channel_B5+(channel_B4/3)).astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2025-34', '2025-41'), 'channel_B'] = (channel_B6+(channel_B4/3)).astype(int)

df_media_shift.loc[df_media_shift['KW'].between('2023-10', '2023-17'), 'channel_C'] = channel_C1.astype(int)+channel_C3.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2023-35', '2023-42'), 'channel_C'] = channel_C2.astype(int)+channel_C5.astype(int)
#df_media_shift.loc[df_media_shift['KW'].between('2024-12', '2024-19'), 'channel_C'] = channel_C3.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2024-32', '2024-39'), 'channel_C'] = channel_C4.astype(int)
df_media_shift.loc[df_media_shift['KW'].between('2025-10', '2025-17'), 'channel_C'] = channel_C6.astype(int)
#df_media_shift.loc[df_media_shift['KW'].between('2025-33', '2025-40'), 'channel_C'] = channel_C6.astype(int)

#scaling media variables
scaler_geo_dict={}
for media_channel in media_vars:
    scaler = MinMaxScaler()
    df_media_shift[f'{media_channel}_sc'] = scaler.fit_transform(df_media_shift[[media_channel]]).flatten()
    scaler_geo_dict[media_channel] = scaler
    
df_media_shift['channel_A_tf'] = saturate(geometric_adstock(df_media_shift['channel_A_sc'], ads_A).eval(),sat_A).eval()
df_media_shift['channel_B_tf'] = saturate(geometric_adstock(df_media_shift['channel_B_sc'], ads_B).eval(),sat_B).eval()
df_media_shift['channel_C_tf'] = saturate(geometric_adstock(df_media_shift['channel_C_sc'], ads_C).eval(),sat_C).eval()
  
#calculate contribution for media dynamic
i=0
for media_channel in media_vars:
    scaler = scaler_geo_dict[media_channel]
    df_media_shift[f'{media_channel}_contr'] = np.array(scaler.inverse_transform(df_media_shift[[f'{media_channel}_tf']])*coefficients[i]).flatten().round()
    i=i+1

df_media_shift['distribution_contr']=df_media_shift['distribution']*5000
df_media_shift['season_contr']=df_media_shift['season']*1500

df_media_shift['umsatz']=df_media_shift[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_shift=df_media_shift[['date','KW','KW_year','Marke','umsatz','distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr','distribution','season','channel_A','channel_B','channel_C']]


#media with constant
df_media_cons=df

df_media_cons['channel_A'] = 0
df_media_cons['channel_B'] = 0
df_media_cons['channel_C'] = 0

df_media_cons.loc[df_media_cons['KW'].between('2023-01', '2023-26'), 'channel_A'] = (channel_A1.sum()/26).astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2023-27', '2023-52'), 'channel_A'] = (channel_A2.sum()/26).astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2024-01', '2024-26'), 'channel_A'] = (channel_A3.sum()/26).astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2024-27', '2024-52'), 'channel_A'] = (channel_A4.sum()/26).astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2025-01', '2025-26'), 'channel_A'] = (channel_A5.sum()/26).astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2025-27', '2025-52'), 'channel_A'] = (channel_A6.sum()/26).astype(int)

df_media_cons.loc[df_media_cons['KW'].between('2023-10', '2023-17'), 'channel_B'] = channel_B1.astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2023-32', '2023-39'), 'channel_B'] = channel_B2.astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2024-10', '2024-17'), 'channel_B'] = (channel_B3+(channel_B4/3)).astype(int)
#df_media_cons.loc[df_media_cons['KW'].between('2024-36', '2024-43'), 'channel_B'] = channel_B4.astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2025-08', '2025-15'), 'channel_B'] = (channel_B5+(channel_B4/3)).astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2025-34', '2025-41'), 'channel_B'] = (channel_B6+(channel_B4/3)).astype(int)

df_media_cons.loc[df_media_cons['KW'].between('2023-10', '2023-17'), 'channel_C'] = channel_C1.astype(int)+channel_C3.astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2023-35', '2023-42'), 'channel_C'] = channel_C2.astype(int)+channel_C5.astype(int)
#df_media_cons.loc[df_media_cons['KW'].between('2024-12', '2024-19'), 'channel_C'] = channel_C3.astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2024-32', '2024-39'), 'channel_C'] = channel_C4.astype(int)
df_media_cons.loc[df_media_cons['KW'].between('2025-10', '2025-17'), 'channel_C'] = channel_C6.astype(int)
#df_media_cons.loc[df_media_cons['KW'].between('2025-33', '2025-40'), 'channel_C'] = channel_C6.astype(int)

#scaling media variables
scaler_geo_dict={}
for media_channel in media_vars:
    scaler = MinMaxScaler()
    df_media_cons[f'{media_channel}_sc'] = scaler.fit_transform(df_media_cons[[media_channel]]).flatten()
    scaler_geo_dict[media_channel] = scaler
    
df_media_cons['channel_A_tf'] = saturate(geometric_adstock(df_media_cons['channel_A_sc'], ads_A).eval(),sat_A).eval()
df_media_cons['channel_B_tf'] = saturate(geometric_adstock(df_media_cons['channel_B_sc'], ads_B).eval(),sat_B).eval()
df_media_cons['channel_C_tf'] = saturate(geometric_adstock(df_media_cons['channel_C_sc'], ads_C).eval(),sat_C).eval()
  
#calculate contribution for media dynamic
i=0
for media_channel in media_vars:
    scaler = scaler_geo_dict[media_channel]
    df_media_cons[f'{media_channel}_contr'] = np.array(scaler.inverse_transform(df_media_cons[[f'{media_channel}_tf']])*coefficients[i]).flatten().round()
    i=i+1

df_media_cons['distribution_contr']=df_media_cons['distribution']*5000
df_media_cons['season_contr']=df_media_cons['season']*1500

df_media_cons['umsatz']=df_media_cons[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_cons=df_media_cons[['date','KW','KW_year','Marke','umsatz','distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr','distribution','season','channel_A','channel_B','channel_C']]



# Optional: Save as CSV
df_media_even.to_csv('df_media_even.csv', index=False)
print("\nDataset was saved as 'df_media_even.csv'.")

df_media_dyn.to_csv('df_media_dynamic.csv', index=False)
print("\nDataset was saved as 'df_media_dynamic.csv'.")

df_media_shift.to_csv('df_media_time_shift.csv', index=False)
print("\nDataset was saved as 'df_media_time_shift.csv'.")

df_media_cons.to_csv('df_media_constant.csv', index=False)
print("\nDataset was saved as 'df_media_constant.csv'.")




#descriptives---------------------------------------------------------
plt.plot(df['KW'],df['distribution'])

plt.plot(df['KW'],df['season'])

plt.plot(wetter['KW'],wetter['TEMPERATUR_MITTEL'])

plt.plot(df_media_even['KW'],df_media_even['umsatz'])
plt.plot(df_media_dyn['KW'],df_media_dyn['umsatz'])

#ROI even
(df_media_even['channel_A_contr'].sum() / df_media_even['channel_A'].sum()).round(2)
(df_media_even['channel_B_contr'].sum() / df_media_even['channel_B'].sum()).round(2)
(df_media_even['channel_C_contr'].sum() / df_media_even['channel_C'].sum()).round(2)

#ROI dyv
(df_media_dyn['channel_A_contr'].sum() / df_media_dyn['channel_A'].sum()).round(2)
(df_media_dyn['channel_B_contr'].sum() / df_media_dyn['channel_B'].sum()).round(2)
(df_media_dyn['channel_C_contr'].sum() / df_media_dyn['channel_C'].sum()).round(2)

#ROI time shift
(df_media_shift['channel_A_contr'].sum() / df_media_shift['channel_A'].sum()).round(2)
(df_media_shift['channel_B_contr'].sum() / df_media_shift['channel_B'].sum()).round(2)
(df_media_shift['channel_C_contr'].sum() / df_media_shift['channel_C'].sum()).round(2)

#ROI constant
(df_media_cons['channel_A_contr'].sum() / df_media_cons['channel_A'].sum()).round(2)
(df_media_cons['channel_B_contr'].sum() / df_media_cons['channel_B'].sum()).round(2)
(df_media_cons['channel_C_contr'].sum() / df_media_cons['channel_C'].sum()).round(2)

df_media_even[['channel_A','channel_B','channel_C']].sum().sum()
df_media_dyn[['channel_A','channel_B','channel_C']].sum().sum()
df_media_shift[['channel_A','channel_B','channel_C']].sum().sum()
df_media_cons[['channel_A','channel_B','channel_C']].sum().sum()


#plot area chart
ax = (df_media_even[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )

#plot area chart
ax = (df_media_dyn[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )

#plot area chart
ax = (df_media_shift[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )

#plot area chart
ax = (df_media_cons[['distribution_contr','season_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )




#### add new features #####
#read in data from first runs
df_media_even=pd.read_csv('df_media_even.csv')
df_media_dyn=pd.read_csv('df_media_dynamic.csv')
df_media_shift=pd.read_csv('df_media_time_shift.csv')
df_media_cons=pd.read_csv('df_media_constant.csv')


#additional variables
def add_price_variable(df, kw_column='KW', base_price=10, random_seed=None):
      """
      Adds a price variable with seasonal effects and a trend.
      
      Parameter:
      - df: DataFrame with a calendar week column
      - kw_column: Name of the column containing weeks in 'YYYY-WW' format
      - base_price: Base price around which fluctuations occur (default: 10 euros)
      - random_seed: Seed for reproducibility (optional)
      
      Returns:
      - DataFrame with new columns 'price' and 'price_transformed'
      """
      if random_seed is not None:
          np.random.seed(random_seed)

      df = df.copy()

      # Parse year and week
      df['_year'] = df[kw_column].str[:4].astype(int)
      df['_week'] = df[kw_column].str[-2:].astype(int)

      # Baseline price
      price = np.ones(len(df)) * base_price

      # 1. Annual trend (decreasing): -0.5 euros per year
      years_since_start = df['_year'] - df['_year'].min()
      trend = -0.5 * years_since_start

      # 2. Seasonal components based on the week
      seasonal = np.zeros(len(df))

      for idx, week in enumerate(df['_week']):
          # Spring (weeks 10–20, approx. March–May): -2 euros
          if 10 <= week <= 20:
              seasonal[idx] = -2.0

          # Summer (weeks 21–35, approx. May–August): normal
          elif 21 <= week <= 35:
              seasonal[idx] = 0

          # Autumn (weeks 36–45, approx. September–November): -1.5 euros
          elif 36 <= week <= 45:
              seasonal[idx] = -1.5

          # December (weeks 48–52): +3 euros (Christmas business)
          elif week >= 48:
              seasonal[idx] = 3.0

          # Winter (weeks 1–9, approx. January–February): slightly elevated
          else:
              seasonal[idx] = 0.5

      # 3. Random fluctuation: ±20% of the base price
      random_variation = np.random.uniform(-0.2 * base_price, 0.2 * base_price, len(df))

      # Final Price Calculation
      df['price'] = price + trend + seasonal + random_variation

      # Ensure that prices do not become negative
      df['price'] = np.maximum(df['price'], 1.0)

      # 4. Transformed variable (inverted so higher price = negative effect)
      # Option A: Negative of the price
      # df['price_transformed'] = -df['price']

      # Option B: Maximum minus price (so that higher values ​​= lower price)
      max_price = df['price'].max()
      df['price_transformed'] = max_price - df['price']

      # Option C: Inverse price (1/price)
      # df['price_transformed'] = 1 / df['price']

      # Cleanup of temporary columns
      df.drop(['_year', '_week'], axis=1, inplace=True)

      return df
  
df_media_even = add_price_variable(df_media_even, random_seed=42)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(df_media_even['price'])
plt.title('Price history over time')
plt.xlabel('week')
plt.ylabel('price (€)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df_media_even['price_transformed'])
plt.title('Transformed price variable')
plt.xlabel('week')
plt.ylabel('Transformed value')
plt.grid(True)

plt.tight_layout()
plt.show()


df_media_dyn['price'] = df_media_even['price']
df_media_shift['price'] = df_media_even['price']
df_media_cons['price'] = df_media_even['price']
df_media_dyn['price_transformed'] = df_media_even['price_transformed']
df_media_shift['price_transformed'] = df_media_even['price_transformed']
df_media_cons['price_transformed'] = df_media_even['price_transformed']

df_media_even['price_contr'] = df_media_even['price_transformed'] * 37500
df_media_dyn['price_contr'] = df_media_dyn['price_transformed'] * 37500
df_media_shift['price_contr'] = df_media_shift['price_transformed'] * 37500
df_media_cons['price_contr'] = df_media_cons['price_transformed'] * 37500



def add_competition_variable_extended(df, week_column='KW_year', 
                                     baseline_competition=True, 
                                     random_seed=None):
      """
      Adds a competition variable with peak periods.
      
      Parameter:
      - df: DataFrame with a week number column
      - week_column: Name of the column containing the week number within the year
      - baseline_competition: If True, other weeks have low baseline values
      - random_seed: Seed for reproducibility
      
      Returns:
      - DataFrame with a new 'competition' column
      """
      if random_seed is not None:
          np.random.seed(random_seed)

      df = df.copy()

      # Baseline values ​​for all other weeks (if desired)
      if baseline_competition:
          # Low competition outside of peak times
          df['competition'] = np.random.uniform(20000, 50000, len(df))
      else:
          df['competition'] = 0

      # Peak 1: Weeks 14–23 (spring/summer)
      mask_spring = (df[week_column] >= 14) & (df[week_column] <= 23)
      n_spring = mask_spring.sum()
      df.loc[mask_spring, 'competition'] = np.random.uniform(100000, 150000, n_spring)

      # Peak 2: Weeks 38–47 (Autumn)
      mask_autumn = (df[week_column] >= 38) & (df[week_column] <= 47)
      n_autumn = mask_autumn.sum()
      df.loc[mask_autumn, 'competition'] = np.random.uniform(130000, 180000, n_autumn)

      return df
  
df_media_even = add_competition_variable_extended(df_media_even, baseline_competition=True, random_seed=42)

# Visualize competition over time
plt.figure(figsize=(14, 5))
plt.plot(df_media_even['competition'])
plt.axhline(y=100000, color='r', linestyle='--', alpha=0.3, label='Spring Range')
plt.axhline(y=150000, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=130000, color='g', linestyle='--', alpha=0.3, label='Autumn Range')
plt.axhline(y=180000, color='g', linestyle='--', alpha=0.3)
plt.title('Competition over Time')
plt.xlabel('week')
plt.ylabel('Competition Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


df_media_dyn['competition'] = df_media_even['competition']
df_media_shift['competition'] = df_media_even['competition']
df_media_cons['competition'] = df_media_even['competition']
df_media_even['competition_contr'] = df_media_even['competition'] *0.43
df_media_dyn['competition_contr'] = df_media_dyn['competition']*0.43
df_media_shift['competition_contr'] = df_media_shift['competition']*0.43
df_media_cons['competition_contr'] = df_media_cons['competition']*0.43


# segment width
df_media_even['segment'] = 0.8
df_media_even.loc[df_media_even['KW'].between('2023-39', '2023-49'), 'segment'] = 1.0
df_media_even.loc[df_media_even['KW'].between('2023-50', '2024-19'), 'segment'] = 0.7
df_media_even.loc[df_media_even['KW'].between('2024-20', '2024-37'), 'segment'] = 0.6
df_media_even.loc[df_media_even['KW'].between('2024-38', '2025-13'), 'segment'] = 0.9
df_media_even.loc[df_media_even['KW'].between('2025-14', '2025-40'), 'segment'] = 0.8
df_media_even.loc[df_media_even['KW'].between('2025-41', '2026-01'), 'segment'] = 0.5

df_media_even['segment_contr'] = df_media_even['segment']*120000

df_media_dyn['segment'] = df_media_even['segment']
df_media_shift['segment'] = df_media_even['segment']
df_media_cons['segment'] = df_media_even['segment']
df_media_dyn['segment_contr'] = df_media_even['segment_contr']
df_media_shift['segment_contr'] = df_media_even['segment_contr']
df_media_cons['segment_contr'] = df_media_even['segment_contr']



# revenue calculation
df_media_even['revenue']=df_media_even[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_dyn['revenue']=df_media_dyn[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_shift['revenue']=df_media_shift[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)
df_media_cons['revenue']=df_media_cons[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']].sum(axis=1)




# Optional: Save as CSV
df_media_even.to_csv('df_media_even.csv', index=False)
print("\nThe dataset was saved as 'df_media_even.csv'.")

df_media_dyn.to_csv('df_media_dynamic.csv', index=False)
print("\nDataset was saved as 'df_media_dynamic.csv'.")

df_media_shift.to_csv('df_media_time_shift.csv', index=False)
print("\nThe dataset was saved as 'df_media_time_shift.csv'.")

df_media_cons.to_csv('df_media_constant.csv', index=False)
print("\nThe dataset was saved as 'df_media_constant.csv'.")




## decomposition charts
ax = (df_media_even[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )

ax = (df_media_dyn[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )

ax = (df_media_shift[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )

ax = (df_media_cons[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr','channel_A_contr','channel_B_contr','channel_C_contr']]
.plot.area(figsize=(16, 10), linewidth=1, title='Contribution per variable', ylabel='Target', xlabel='week'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(
handles[::-1], labels[::-1],
title='Variables', loc="center left",
bbox_to_anchor=(1.01, 0.5) )


## check contributions sums
media_sum=df_media_even[['channel_A_contr','channel_B_contr','channel_C_contr']].sum().sum()
other_sum=df_media_even[['distribution_contr','season_contr','price_contr','competition_contr','segment_contr']].sum().sum()

round(media_sum/(media_sum+other_sum)*100,1)

media_sum=df_media_even[['channel_A_contr','channel_B_contr','channel_C_contr']].sum().sum()
other_sum=df_media_even[['distribution_contr','season_contr']].sum().sum()

round(media_sum/(media_sum+other_sum)*100,1)