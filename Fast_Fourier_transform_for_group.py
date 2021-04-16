'''
April 13, 2021
Title: Implementation of regression techniques on time-series data to generate future predictions
Module: Fast Fourier Transform Forecasting Model (FFT) model for group resource prediction


Reference for freq parameter for time series model
Alias    Description
B        business day frequency
C        custom business day frequency
D        calendar day frequency
W        weekly frequency
M        month end frequency
SM       semi-month end frequency (15th and end of month)
BM       business month end frequency
CBM      custom business month end frequency
MS       month start frequency
SMS      semi-month start frequency (1st and 15th)
BMS      business month start frequency
CBMS     custom business month start frequency
Q        quarter end frequency
BQ       business quarter end frequency
QS       quarter start frequency
BQS      business quarter start frequency
A, Y     year end frequency
BA, BY   business year end frequency
AS, YS   year start frequency
BAS, BYS business year start frequency
BH       business hour frequency
H        hourly frequency
T, min   minutely frequency
S        secondly frequency
L, ms    milliseconds
U, us    microseconds
N        nanoseconds
'''
#imports
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import FFT
from darts.dataprocessing.transformers import MissingValuesFiller
import matplotlib.pyplot as plt

from darts.metrics import mape
from darts.utils.missing_values import fill_missing_values

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

from pre_processing import read_file, clean_dataframe, format_datetime

group_name = 'group2'
df = read_file('./data.csv', 'group', group_name)
df = clean_dataframe(df)
df = format_datetime(df)

#summary of the dataset
print(df.head(5))
print(df.info())

#select the time series variable
ts_var = 'CPU_Used'
time_df = df.filter(['date', ts_var])

#convert variable from object dtype to numeric dtype
time_df[ts_var] = pd.to_numeric(time_df[ts_var], errors='coerce')

#remove duplicates
time_df.sort_values("date", inplace = True) 
time_df.drop_duplicates(subset ="date", keep = False, inplace = True)

#generate time series using darts
series = TimeSeries.from_dataframe(time_df, 'date', ts_var, freq='T')

#treat missing values
filler = MissingValuesFiller()
series = filler.transform(series)

#training and testing dataset
train, val = series.split_after(pd.Timestamp('2019-10-23 19:41:50'))

#FFT model
model = FFT(required_matches=set(), nr_freqs_to_keep=None)
model.fit(train)
pred_val = model.predict(len(val))

#Evaluation metrics
series.plot(label='actual')
pred_val.plot(label='forecast', lw=3)
plt.legend()
print("MAPE:", mape(pred_val, val))

