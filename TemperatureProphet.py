#Machlan Pettersen
#This system uses the Neural Prophet package to create a neural network that
#can predict temperature swings in the upcoming day.

#install neural prophet using: pip install neuralprophet
#import neuralprophet, pandas, and matplotlib
from neuralprophet import NeuralProphet
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#read in weather data using pandas.
#path_string = r'' Enter path string here.
df = pd.read_excel(path_string)
print(df.columns)

#convert dates from arbitrary objects to datetime objects.
df['Date time'] = pd.to_datetime(df['Date time'])
#printing dataframe head and types.
print(df.head())
print(df.dtypes)

#convert celsius into fahrenheit so I can understand the data.
df['Temperature'] = df['Temperature'].apply(lambda x: x * 1.8 + 32)

#create new df column to represent Hours.
df['Hour'] = df['Date time'].apply(lambda x: x.hour)
#use only the first 6 months to reduce CPU strain.
df = df[df['Hour'] <= 24]

#prepare data to be passed to neural prophet
#select the date time value and temperature value from the dataframe.
prophet_data = df[['Date time','Temperature']]
#remove NaN values from each column.
prophet_data.dropna(inplace = True)
#declare the two input columns to be passed to the model. 'ds' represents time, and 'y' represents temperature.
prophet_data.columns = ['ds','y']

#create a new instance of a neural prophet model.
#disable yearly and weekly seasonality for the sake of freeing up CPU.
prophet = NeuralProphet(yearly_seasonality=False, weekly_seasonality = False)


#split the data into a training set and validation set using an 80:20 split
prophet_train, prophet_test = prophet.split_df(prophet_data, freq='15T', valid_p = 0.2)


#fit the model using a 15-minute interval.
## SIGNIFICANT CPU STRAIN ##
metrics = prophet.fit(prophet_train, freq = '15T', validation_df= prophet_test)


#create a prediction using 96 iterations to represent 96 15-minute intervals within each day.
future = prophet.make_future_dataframe(prophet_data, periods = 96)


#plotting metrics.
forecast = prophet.predict(future)
plot_one = prophet.plot(forecast)
fig_components = prophet.plot_components(forecast)
fig_model = prophet.plot_parameters(forecast)
plt.show()




#gathering data from the prediction (yhat1), and saving it in a list.
temp_prediction_list = []

for i, value in enumerate(forecast['yhat1']):
    temp_prediction_list.append(value)
previous_temp = temp_prediction_list[0]

#scrubbing the prediction list for rapid changes in temperature (over 0.7 degrees F within 15 minutes).
swing_counter = 0
for y in range(len(temp_prediction_list)):
    current_temp = temp_prediction_list[y]
    difference = current_temp - previous_temp
    if abs(difference) > 0.7:
        swing_counter += 1
    previous_temp = current_temp
print(f"{swing_counter} rapid temperature changes noticed within prediction dataframe")

#using numpy to find the derivative of temperature with respect to time.
derivative_list = []
dx = 15
dy = np.diff(temp_prediction_list)
derivative_list.append(dy/dx)

#calculating the critical points within the daily temperature swing.
swing_indices = [i for i, x in enumerate(np.concatenate(derivative_list)) if x > 0]


swing_begin_index = swing_indices[0]
swing_begin_temp = temp_prediction_list[swing_begin_index]

swing_end_index = swing_indices[-1]
swing_end_temp = temp_prediction_list[swing_end_index]

swing_time = len(swing_indices)*15
print(f'swing begin index = {swing_begin_index}, at a temperature of {swing_begin_temp: .2f}F')
print(f'swing end index = {swing_end_index}, at a temperature of {swing_end_temp: .2f}F')
print(f'this swing occurs over {swing_time} minutes')

## COOLING LOAD ASSUMPTIONS ##
# 1. Target temperature (T_in) = 68 degrees F
# 2. Building surface area (A) = 500 m^2
# 3. U-value of insulation (U) = 0.3 W/m^2*k
# 4. Q = U * A * (T_out - T_in)

#defining cooling load constants.
T_in = (68 - 32) * (5/9) #converting to C
A = 500
U = 0.3

#get a list of all temperature intervals that require cooling.
cooling_intervals = temp_prediction_list
for t in temp_prediction_list:
    if t < 68:
        t = 0
print(cooling_intervals)

#find how much cooling needs to be applied every 15 minutes.
# cooling_load_required = []
# for T_out in cooling_intervals:
#     Q = U * A * (T_out - T_in) # in WATTS
#     cooling_load_required.append(Q)
#
# #sum cooling load, and plot!
# daily_cooling = sum(cooling_load_required)
# print(daily_cooling)
# daily_15t = np.linspace(0,1440,96)
# plt.plot(daily_15t, cooling_load_required)
# plt.xlabel("Time (minutes)")
# plt.ylabel("Cooling load required (Watts)")
# plt.show()
