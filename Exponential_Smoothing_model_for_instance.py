'''April 13 , 2021
Title: Implementation of regression techniques on time-series data to generate future predictions
Module: Exponential Smoothing model for instance resource prediction


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
from darts.models import ExponentialSmoothing
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
import matplotlib.pyplot as plt
from darts.metrics import mape

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


from pre_processing import read_file, clean_dataframe, format_datetime

instance_name = '1b6ffb4a-b7bc-48d0-ab60-b43f64b7c6f4'
df = read_file('./data.csv', 'instance', instance_name)
df = clean_dataframe(df)
df = format_datetime(df)

#summary of the dataset
print(df.head(5))
print(df.info())

#select the time series variable
ts_var = 'Memory_Used'
time_df = df.filter(['date', ts_var])

#convert variable from object dtype to numeric dtype
time_df[ts_var] = pd.to_numeric(time_df[ts_var], errors='coerce')

#generate time series using darts
series = TimeSeries.from_dataframe(time_df, 'date', ts_var, freq='S')

#treat missing values
filler = MissingValuesFiller()
series = filler.transform(series)

#scale the values
scaler = Scaler()
rescaled = scaler.fit_transform(series)

#training and testing dataset
train, val = rescaled.split_after(pd.Timestamp('2020-01-23 19:41:50'))

#Exponential smoothing model
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))

#Evaluation metrics
rescaled.plot(label='actual')
prediction.plot(label='forecast', lw=3)
plt.legend()
print("MAPE:", mape(prediction, val))

