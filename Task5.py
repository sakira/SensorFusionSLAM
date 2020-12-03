#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:48:01 2020
Task 5
@author: hassans4
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CAMERA_COLUMNS = [
    'timestamp', 'qr_code', 'y1', 'y2', 'width', 'height', 'distance', 'attitude'
 
]


# robots global position
P_r_x = 60
P_r_y = 39
r_psi = 90

f = 6094.36
f = 539.1304347826087



#%% Read the task 5 data

dataset = 'dataset1'
data_path = '../project_data/' + dataset + '/data/task5/'
data = pd.read_csv(data_path + 'camera_localization_task5.csv', header=None)

data.columns = CAMERA_COLUMNS
print(data)

# add delta t column
data['delta_t'] = data.timestamp - data.timestamp[0]

# read the global positions of qr codes
all_gp_qr_codes = pd.read_csv('../project_data/qr_code_position_in_global_coordinate.csv')
all_gp_qr_codes.rename(columns=lambda x: x.strip(), inplace=True)

# Take one measurement
index = data.timestamp[0]
# Y = data[data.timestamp==index, ['distance', 'attitude']].to_numpy()
Y = data[data.timestamp==index][['distance', 'attitude']]
Y = np.reshape(Y.values, (Y.shape[0]*Y.shape[1], 1))




unique_qr_codes = data[data.timestamp==index].qr_code.unique()
gp_qr_codes = all_gp_qr_codes[all_gp_qr_codes.qr_code.isin(unique_qr_codes)]

# selected columns
gp_qr_codes = gp_qr_codes[['mid_point_x_cm', 'mid_point_y_cm']]
gp_qr_codes
gp_qr_codes = list(zip(*map(gp_qr_codes.get, gp_qr_codes)))
s_i = list(zip(*gp_qr_codes))



#%% Define measurement functions and Jacobians
# compute distance given two coordinates
def distance(px, py, sx, sy):
    return np.sqrt( (sx - px)**2 + (sy - py)**2 )
# compute angle between two coordinates
def angle(px, py, sx, sy, psi):
    return np.arctan2( (sx - px), (sy - py)) - psi

# the measurement model
def g_x(p_i, s_i):
    '''
    p_i: current position of the robot px, py, psi
    s_i: global positions of the detected QR codes sx, sy
    '''
    px, py, psi = p_i
    sx, sy = s_i
    n_sensors = len(sx) #s_i.shape[0]

    # print(p_i, s_i)    
    
    # variables
    y_dist = np.zeros((n_sensors, 1))
    y_phi = np.zeros((n_sensors, 1))
    
    
    # for each sensor calculate the distance and angle
    for i in range(0, n_sensors):
        y_dist[i] = distance(px, py, sx[i], sy[i])
        y_phi[i] = angle(px, py, sx[i], sy[i], psi) 
        # convert phi to radian
        y_phi[i] = y_phi[i] * np.pi/180
        
        
        
    # # fancy stuffs
    # res = {'distance': y_dist, 'phi': y_phi}
    # df = pd.DataFrame(data=y_dist, columns=['dx'])
    # df2 = pd.DataFrame(data=y_phi, columns=['phi'])
    
    # df = df.append(df2)
    
    # model = df.values
    # print(model)
    
    res = {'distance': y_dist.flatten(), 'phi': y_phi.flatten()}
    res = pd.DataFrame.from_dict(res)
    # print(res)
    
    model = np.reshape(res.values, (res.shape[0] * res.shape[1], 1))
    
    return model
    
    return y_dist, y_phi


#%% Test the model
# p_i = [0., 0., 0.]

# model = g_x(p_i, s_i)
# print(model)
# Jx = Y - model
# jacobian_model = G_x(p_i, s_i)
# # print(jacobian_model)
# R_one_diag = np.array([10, 2])
# R_one_diag
# # R = np.kron(np.diag(R_one_diag))
# # R
# num_qr_codes = unique_qr_codes.shape[0]
# R = np.diag(np.kron(np.ones(num_qr_codes),R_one_diag))
# R.shape


# LR = np.linalg.cholesky(R)
# Jxx = np.sum((np.linalg.solve(R, Jx))**2)

# xi = x0
# temp = G_x(xi, s_i).T@R@G_x(xi, s_i)
# temp

# temp2 = Y - g_x(xi, s_i)
# temp2

# temp3 = G_x(xi, s_i)
# temp3
    


#%%

# The Jacobian for the measurement model
def G_x(p_i, s_i):
    '''
    p_i: current position of the robot px, py, psi
    s_i: global positions of the detected QR codes sx, sy
    '''
    px, py, psi = p_i
    sx, sy = s_i
    n_sensors = len(sx) #s_i.shape[0]
    
    
    # variables
    G_y_dist = np.zeros((n_sensors, 2)) # x = [px, py]
    G_y_phi = np.zeros((n_sensors, 3)) # x = [px, py, psi]
    
    
    # for each sensor calculate the distance and angle
    for i in range(0, n_sensors):
        dist = distance(px, py, sx[i], sy[i])
        
        G_y_dist[i] = np.array([-(sx[i]-px)/dist, -(sy[i]-py)/dist])
        G_y_phi[i] = np.array([-(sy[i]-py)/dist**2, -(sx[i]-px)/dist**2, -1])
        
        # convert phi to radian
        G_y_phi[i] = G_y_phi[i] * np.pi/180
        
        
    # fancy stuffs
    resGx = {'G_distance': G_y_dist, 'G_phi': G_y_phi}
    

    
    df = pd.DataFrame(data=G_y_dist, columns=['dxpx', 'dxpy'])
    df['dxpsi'] = 0.0
    
    
    df2 = pd.DataFrame(data=G_y_phi, columns=['dxpx', 'dxpy', 'dxpsi'])
    
    # print(df, df2)
    # print(df.stack())
    
    model = np.hstack([df.values, df2.values]).reshape(-1, df.shape[1])
    # print(model)
    
    
    
    # df = df.append(df2)
    
    
    
    
    
    # model = df.values
        
    return model

#%% Define the cost function
def Jwls(y, x_i, s_i, R):
    Jx = y - g_x(x_i, s_i)
    # Jx = Jx.T@Rinv@Jx
    LR = np.linalg.cholesky(R)
    Jx = np.sum((np.linalg.solve(R, Jx))**2)
    return Jx

#%% 
def ls_grid_search(xi, dx, Ng, s_i, Rinv, y):
    
    opt_gamma = 1
    Jopt = Jwls(y, xi, s_i, Rinv)
    
    for j in range(1, Ng+1):
        gamma = j/Ng
        
        xi_new = xi + gamma * dx

        Jnew = Jwls(y, xi_new, s_i, Rinv)
        
        if (Jnew < Jopt):
            
            opt_gamma = gamma
            J = Jnew
    
    return opt_gamma



#%% Test algorithm

# initialization parameters
x0 = np.array([0, 0, 0])
R_one_diag = np.array([1,1])
num_qr_codes = unique_qr_codes.shape[0]
R = np.diag(np.kron(np.ones(num_qr_codes),R_one_diag))
LR = np.linalg.cholesky(R).T
Rinv = np.diag(1/np.diag(R))
Rinv.shape

number_of_iterations = 100
Ng = 100 # number of grid points for gamma
# s_i = list(zip(*gp_qr_codes))



xi = x0
# =============================================================================
# dx = np.linalg.inv(G_x(xi, s_i).T@Rinv@G_x(xi, s_i))@(G_x(xi, s_i).T@Rinv)@(y - g_x(xi, s_i))
# =============================================================================

x_history = []
for i in range(0, number_of_iterations):
    
    # udpate direction
    # dx = np.linalg.inv(G_x(xi, s_i).T@Rinv@G_x(xi, s_i))@(G_x(xi, s_i).T@Rinv)@(y - g_x(xi, s_i))
    # dx = np.linalg.solve( G_x(xi, s_i).T@LR@G_x(xi, s_i), 
    #                      (G_x(xi, s_i).T@LR)@(Y - g_x(xi, s_i)) )
    
    dx = np.linalg.solve( G_x(xi, s_i).T@Rinv@G_x(xi, s_i), G_x(xi, s_i).T@Rinv@(Y - g_x(xi, s_i)) )
                         
    #need to squeeze the shape of dx to match xi
    dx = np.squeeze(dx)
    print(dx, xi)
    
    # find optimal gamma
    gamma = ls_grid_search(xi, dx, Ng, s_i, R, Y)


    # update
    xi = xi + gamma * dx
    
    # save
    x_history.append(xi)


# np.abs(angle_1) + alpha*180/np.pi +180
angle_to_degree = xi[2] #* 180/np.pi

print('Estimated values:')
print('xi = {}'.format(xi),' iteration = '  , i, ' gamma = ', gamma, 'psi = {}'.format(angle_to_degree))
print('True values:')
print('P_r_x = ', P_r_x,' P_r_y = ',  P_r_y, 'r_psi', r_psi)




