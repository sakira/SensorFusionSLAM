#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:11:46 2020

@author: hassans4
"""

#%% Task 2: Experiments

import numpy as np
from sensors import Sensor
from sensors import IMU_COLUMNS


import matplotlib.pyplot as plt

#%%

GRAVITY = 1 # 1g = 9.8 m/s^2

dataset = 'dataset2'

data_path = '../project_data/' + dataset + '/data/task2/'
filename = data_path + 'imu_calibration_task2.csv'

sensor = Sensor(filename)

sensor.plot_gyroscope()
sensor.plot_magnetometer()
sensor.plot_accelerometer()

# save the values
path_to_store = 'results/' + dataset + '/task2/'

# save plots
filename = path_to_store + 'accelerometer.pdf'
sensor.save_plot_accelerometer(filename)
# save plots
filename = path_to_store + 'gyroscope.pdf'
sensor.save_plot_gyroscope(filename)
# save plots
filename = path_to_store + 'magnetometer.pdf'
sensor.save_plot_magnetometer(filename)




print( sensor.compute_total_time()) # ~3 min
#%%
cutoff_value = 70 # 80 sec window
sensor.data['diff_timestamp'] = sensor.data.timestamp - sensor.data.timestamp[0]
threshold = 0.4

#%% For acc_z
start_index = 0
end_index = cutoff_value

az = sensor.data.loc[(sensor.data.diff_timestamp >= start_index) & (sensor.data.diff_timestamp <= end_index)]['acc_z']
az.plot()
az_up = az[az>threshold]
az_up.plot()
az_down = az[az<-threshold]
az_down.plot()
kz = (az_up.mean() - az_down.mean())/(2.0*GRAVITY)
bz = (az_up.mean() + az_down.mean())/2.0
kz2 = (az_up.mean() - az_down.mean())/(2.0)
print('kz = ', kz)
print('bz = ', bz)

#%% For acc_x

start_index = cutoff_value - 20
end_index = 2 * cutoff_value - 20



ax = sensor.data.loc[(sensor.data.diff_timestamp >= start_index) & (sensor.data.diff_timestamp <= end_index)]['acc_x']
ax.plot()
ax_up = ax[ax>threshold+0.4]
ax_up.plot()
ax_down = ax[ax<-threshold-0.2]
ax_down.plot()
kx = (ax_up.mean() - ax_down.mean())/(2.0*GRAVITY)
bx = (ax_up.mean() + ax_down.mean())/2.0
print('kx = ', kx)
print('bx = ', bx)


#%% For acc_y

start_index = 2*cutoff_value - 20
end_index = 3 * cutoff_value

ay = sensor.data.loc[(sensor.data.diff_timestamp >= start_index) & (sensor.data.diff_timestamp <= end_index)]['acc_y']
ay.plot()
ay_up = ay[ay>threshold]
ay_up.plot()
ay_down = ay[ay<-threshold]
ay_down.plot()
ky = (ay_up.mean() - ay_down.mean())/(2.0*GRAVITY)
by = (ay_up.mean() + ay_down.mean())/2.0
print('ky = ', ky)
print('by = ', by)


#%% gain and bias
print('kx = ', kx)
print('ky = ', ky)
print('kz = ', kz)

print('bx = ', bx)
print('by = ', by)
print('bz = ', bz)


#%% Task 2b verify your estimated parameters


# Take the values of sensor data where a_x is positive +1
ax_positive = sensor.data[sensor.data.acc_x > 0.8]
ax_positive[IMU_COLUMNS[1:4]].plot()

#%%
B = np.array([bx, by, bz])
G = np.diag([kx, ky, kz])
As = []
Ao_hat = []

for index, row  in ax_positive.iterrows():
    # print(row[IMU_COLUMNS[1:4]])
    a_s = row[IMU_COLUMNS[1:4]].to_numpy()
    ao_hat = a_s@G - B
    
    As.append(a_s[0])
    Ao_hat.append(ao_hat)
    




#%%
Ao = np.array([1.0, 0.0, 0.0])
error = np.sum((Ao_hat - Ao)**2)


#%% plot the result

plt.figure()
plt.plot(Ao_hat)

plt.plot(As)



#%% Gradient descent Optional Tasks


# read the dataset


data_path = '../project_data/dataset_task2b_optional/task2_optional_data/'

import os
import pandas as pd
files = os.listdir(data_path)

for file in files:
    filename  = data_path + file
    data = pd.read_csv(filename, names=IMU_COLUMNS)
    col = IMU_COLUMNS[1:4]
    data[col].plot()



#%% h(A_k, X)

import scipy.linalg as sla

def h(Ak, K, T, B):
    return sla.solve(K, T@(Ak - B))

N = 14




# =============================================================================
# data = pd.read_csv(data_path + 'camera_localization_task5.csv', header=None)
# 
# data.columns = CAMERA_COLUMNS
# print(data)
# 
# =============================================================================



#%%
# =============================================================================
# # Take all positive +x axis and negative -x axis samples
# start_index = cutoff_value - 20
# end_index = 2 * cutoff_value
# 
# sample_x = sensor.data.loc[
#     (sensor.data.diff_timestamp >= start_index) & (sensor.data.diff_timestamp <= end_index)][IMU_COLUMNS[1:4]]
# 
# print(sample_x)
# 
# # Split positive and negative x axis samples
# sample_x_up = sample_x[sample_x>=0] # beware of threshold
# sample_x_down = sample_x[sample_x<0]
# 
# 
# # Take all positive +z axis and negative -z axis samples
# start_index = 0
# end_index = cutoff_value
# 
# sample_z = sensor.data.loc[
#     (sensor.data.diff_timestamp >= start_index) & (sensor.data.diff_timestamp <= end_index)][IMU_COLUMNS[1:4]]
# 
# print(sample_z)
# 
# # Split positive and negative z axis samples
# sample_z_up = sample_z[sample_z>=0] # beware of threshold
# sample_z_down = sample_z[sample_z<0]
# 
# 
# 
# # Take all positive +y axis and negative -y axis samples
# start_index = 2*cutoff_value - 20
# end_index = 3 * cutoff_value
# 
# sample_y = sensor.data.loc[
#     (sensor.data.diff_timestamp >= start_index) & (sensor.data.diff_timestamp <= end_index)][IMU_COLUMNS[1:4]]
# 
# print(sample_y)
# 
# # Split positive and negative x axis samples
# sample_y_up = sample_y[sample_y>=0] # beware of threshold
# sample_y_down = sample_y[sample_y<0]
# 
# 
# =============================================================================


#%% create the dataset
# =============================================================================
# As_bar = np.zeros((6, 3)) # 6 orientations and 3 axis values
# g = np.ones((6, 1))
# 
# #%% add data
# As_bar[0, :] = np.mean(sample_x_up)
# As_bar[1, :] = np.mean(sample_x_down)
# As_bar[2, :] = np.mean(sample_y_up)
# As_bar[3, :] = np.mean(sample_y_down)
# As_bar[4, :] = np.mean(sample_z_up)
# As_bar[5, :] = np.mean(sample_z_down)
# 
# 
# 
#  
# 
# 
# 
# #%%
# 
# X = np.linalg.norm(np.inv(K@T)@(As_bar - B)) - g
# 
# # cost function
# def cost_function (X, g):
#     return ( )**2
# # gradient descent function
# def gradient_descent_function(X):
#     return (- 2 * (1.1 - np.sin(X)) * np.cos(X))
# def gradient_descent(x0, learning_rate=0.1, precision=0.1, max_iter=100):
#     
#     
#     xi = []
#     loss_history = []
#     cost_history = []
#     
#     x = x0
#     
#     xi.append(x)
#     cost = cost_function(0)
#     cost_history.append(cost)
#     
#     i = 0
#     
#     
#     while i < max_iter : # convergence criteria
#     
#     
#         # update direction
#         gradient = gradient_descent_function(x)
#         # update
#         new_x = x - learning_rate * gradient
#         x = new_x
# 
#         
#         xi.append(x)
#         current_cost = cost_function(new_x)
#         cost_history.append(current_cost)
#         
# 
#         loss_history.append(abs(current_cost - cost))
# 
# #         loss = (cost_history[-1] - current_cost)/cost_history[-1]
# #         cost = current_cost
#         
#     
# #         print(current_cost, precision, loss)
#     
# #         if loss < precision:
#         # convergence criteria
#         if abs(gradient) < precision:
#             break
#         
#         
#         
#         i += 1
# 
#     print('local minima x = ', x, ', current cost = ', current_cost)
#     print('Number of iterations required = ', i)
#     
#     
#     return loss_history, xi, i, cost_history
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # if __main__ == 'main':
# #     pass
# =============================================================================
