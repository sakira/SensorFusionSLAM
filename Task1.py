#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:08:13 2020

@author: hassans4

Task 1 experiment

"""


#%% Task 1: Experiments

from sensors import Sensor
from sensors import IMU_COLUMNS

# Task 1 filename
dataset = 'dataset1'
dataset = 'dataset2'
dataset = 'dataset5'
data_path = '../project_data/' + dataset + '/data/task1/'
filename = data_path + 'imu_reading_task1.csv'

sensor = Sensor(filename)
print(sensor.dts)
print(sensor.avg_sampling_rate)
print(sensor.data)


selected_cols = IMU_COLUMNS[0] # timestamps
selected_cols = IMU_COLUMNS[1:4] # accelerometer
sensor.plot(selected_cols)

selected_cols = IMU_COLUMNS[4:6] # roll and pitch
sensor.plot(selected_cols)

selected_cols = IMU_COLUMNS[6:9] # gyroscope
sensor.plot(selected_cols)

selected_cols = IMU_COLUMNS[9:12] # magnetometer
sensor.plot(selected_cols)

# print average value
selected_cols = IMU_COLUMNS[1:12] # all
avg_data = sensor.avg(selected_cols)
print('Mean')
print(avg_data)

# print variances 
selected_cols = IMU_COLUMNS[1:12] # all
var_data = sensor.variance(selected_cols)
print('Variances')
print(var_data)

# save the values
path_to_store = 'results/' + dataset + '/task1/'
filename = path_to_store + 'avg.txt'
avg_data.to_csv(filename, sep=';', header=False)
filename = path_to_store + 'variance.txt'
var_data.to_csv(filename, sep=';', header=False)

# plot the columns
sensor.plot_gyroscope()
sensor.plot_magnetometer()
sensor.plot_accelerometer()


# save plots
filename = path_to_store + 'accelerometer.pdf'
sensor.save_plot_accelerometer(filename)
# save plots
filename = path_to_store + 'gyroscope.pdf'
sensor.save_plot_gyroscope(filename)
# save plots
filename = path_to_store + 'magnetometer.pdf'
sensor.save_plot_magnetometer(filename)
