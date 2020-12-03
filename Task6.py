#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:40:08 2020
Task 6: Dead reckoning
@author: hassans4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMU_COLUMNS = ['timestamp', 
               'acc_x', 'acc_y', 'acc_z',
               'roll', 'pitch',
               'gyro_x', 'gyro_y', 'gyro_z',
               'magnet_x', 'magnet_y', 'magnet_z'
          ]

CAMERA_COLUMNS = [
    'timestamp', 'qr_code', 'y1', 'y2', 'width', 'height', 'distance', 'attitude'
 
]

ODOMETRY_COLUMNS = [
    'timestamp', 'u1', 'u2'
 
]

# x = 16 cm
# y = 44.6 cm

INITIAL_POSITION = [15.7, 47.5, 90*np.pi/180]
INITIAL_POSITION = [16, 44.6, 90*np.pi/180]
Kv = 6.09/0.3 # cm/s

# read camera data
dataset = 'dataset5'
data_path = '../project_data/' + dataset + '/data/task6/'
camera_data = pd.read_csv(data_path + 'camera_tracking_task6.csv', header=None)
camera_data.columns = CAMERA_COLUMNS

# read motor control 
odometry_data = pd.read_csv(data_path + 'motor_control_tracking_task6.csv', header=None)
odometry_data.columns = ODOMETRY_COLUMNS

# read imu data 
imu_data = pd.read_csv(data_path + 'imu_tracking_task6.csv', header=None)
imu_data.columns = IMU_COLUMNS


# read the global positions of qr codes
all_gp_qr_codes = pd.read_csv('../project_data/qr_code_position_in_global_coordinate.csv')
all_gp_qr_codes.rename(columns=lambda x: x.strip(), inplace=True)


#%% dead reckoning functions

# =============================================================================
# def velocity(u1, u2):
#     return (u1 + u2)*Kv * 0.5
# 
# # dynamic model
# def robot_dynamic(t, x, u):
#     '''
#     Your motion model
#     f(x)
#     '''
#     
#        
#     vx = velocity(u1, u2) * np.cos(psi)
#     vy = velocity(u1, u2) * np.sin(psi)
#     omega_z = psi * np.pi/180 # convert to RADIAN
#     return np.array([vx, vy, omega_z])
# 
# =============================================================================

def get_velocity(u1, u2):
   return (u1 + u2)*Kv * 0.5



def robot_dynamic(t,x,u):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])

def robot_jacobian(t,x,u):
    jac = np.zeros((x.shape[0],x.shape[0]))
    jac[0,2] = -u[0]*np.sin(x[2])
    jac[1,2] = u[0]*np.cos(x[2])
    return jac

# jacobian of dynamic model
def Fx(t,x,u):
    jac = np.zeros((x.shape[0],x.shape[0]))
    jac[0,2] = -u[0]*np.sin(x[2])
    jac[1,2] = u[0]*np.cos(x[2])
    return jac


def euler(f,t_now, x_now,u_now, dt):
    return x_now + f(t_now, x_now,u_now)*dt

def euler_propagate(f,t, x_init,u,dt):
    x_res = np.zeros((t.shape[0],x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0]-1):
        x_res[i+1] = euler(f,t[i], x_res[i],u[i],dt)
    return x_res

def euler_stochatic_propagate(f,t, x_init,u, w, dt):
    x_res = np.zeros((u.shape[0],x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0]-1):
        x_res[i+1] = euler(f,t[i], x_res[i],u[i],dt) +  w[i]
    return x_res

def linearization_propagate(f,jac, t, x_init, u, w, dt):
    x_res = np.zeros((t.shape[0],x_init.shape[0]))
    I = np.eye(x_init.shape[0])
    x_res[0] = x_init
    for i in range(x_res.shape[0]-1):
        A = jac(t[i],x_res[i],u[i])
        F = (I + 0.5*A*dt + A@A*dt*dt/6)*dt #this is approximation
        x_res[i+1] = x_res[i] + F@f(t[i],x_res[i],u[i]) + F@w[i]
    return x_res

def rk4(f,t_now, x_now,u_now, dt):
    k1 = f(t_now, x_now,u_now)
    k2 = f(t_now+0.5*dt,x_now+0.5*dt*k1,u_now)
    k3 = f(t_now+0.5*dt, x_now+0.5*dt*k2,u_now)
    k4 = f(t_now+dt, x_now+dt*k3,u_now)
    return x_now+dt*(k1+2*k2+2*k3+k4)/6


def rk4_propagate(f,t, x_init,u,dt):
    x_res = np.zeros((u.shape[0],x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0]-1):
        x_res[i+1] = rk4(f,t[i], x_res[i],u[i],dt)
    return x_res




#%% 
# get the time
dts_odo = odometry_data['timestamp'].diff().values
dts = imu_data['timestamp'].diff().values
dts[0] = 0.0

# get the gyroscope data
bias = -0.064656 #dataset2
bias = -0.009314 # dataset5
omega = imu_data['gyro_z'].values - bias

#%% Simulate dead reckoning
robot_init = np.array(INITIAL_POSITION)
t_robot = dts
dt_robot = 0.1 #t_robot[1]-t_robot[0]
# dt_robot = np.zeros((t_robot.shape[0], 1))
u_robot = np.zeros((t_robot.shape[0],2))

for i in range(1,len(t_robot)):
    dt = t_robot[i] - t_robot[i-1]
    
    # dt_robot[i] = dt
    
    #v = get_velocity(odometry_data['u1'][i], odometry_data['u2'][i])
    v = Kv
    w_gyro = omega[i] * np.pi/180.0 # TODO: add bias from static IMU, also fix the timestamp??
    
    r_input = [v,w_gyro] #input velocity and gyroscope
    u_robot[i,:] = r_input
        



Q_robot = np.diag([0.001,0.001,0.001])
Q_robot_discrete = dt_robot*Q_robot
q_robot = np.random.randn(t_robot.shape[0],3)@np.linalg.cholesky(Q_robot)
q_robot_discrete = np.sqrt(dt_robot)*np.random.randn(t_robot.shape[0],3)@np.linalg.cholesky(Q_robot_discrete)

x_euler_maruyama = euler_stochatic_propagate(robot_dynamic,t_robot, robot_init, u_robot, q_robot_discrete, dt_robot)
x_robot_RK = rk4_propagate(robot_dynamic, t_robot, robot_init, u_robot, dt_robot)

x_linear = linearization_propagate(robot_dynamic,robot_jacobian, t_robot, robot_init, u_robot, q_robot, dt_robot)


# =============================================================================
# f, ax = plt.subplots(2,2, figsize=(20,20))
# skip = 25
# ax[0,0].plot(t_robot,u_robot[:,0], label='Velocity', linewidth=0.5)
# ax[0,0].set_xlabel('$t$')
# ax[0,0].set_ylabel('$v$')
# ax[0,0].legend()
# ax[0,1].plot(t_robot,u_robot[:,1]/np.pi, label='Gyroscope', linewidth=0.5)
# ax[0,1].set_xlabel('$t$')
# ax[0,1].set_ylabel('$\omega / \pi$')
# ax[0,1].legend()
# ax[1,0].plot(x_euler_maruyama[:,0],x_euler_maruyama[:,1], '-b', label='Robot-position-EM', linewidth=0.5)
# ax[1,0].plot(x_linear[:,0],x_linear[:,1], '-r', label='Robot-position-LIN', linewidth=0.5)
# ax[1,0].plot(x_robot_RK[:,0],x_robot_RK[:,1], '-k', label='Robot-position-RK', linewidth=0.5)
# ax[1,0].quiver(x_euler_maruyama[::skip,0],x_euler_maruyama[::skip,1],
#                np.cos(x_euler_maruyama[::skip,2]),np.sin(x_euler_maruyama[::skip,2]),
#                label='Direction-EM', linewidth=0.5, alpha=0.5, color='blue')
# ax[1,0].quiver(x_linear[::skip,0],x_linear[::skip,1],
#                np.cos(x_linear[::skip,2]),np.sin(x_linear[::skip,2]),
#                label='Direction-LIN', linewidth=0.5, alpha=0.5, color='red')
# ax[1,0].quiver(x_robot_RK[::skip,0],x_robot_RK[::skip,1],
#                np.cos(x_robot_RK[::skip,2]),np.sin(x_robot_RK[::skip,2]),
#                label='Direction-RK', linewidth=0.5, alpha=0.5, color='black')
# ax[1,0].set_xlabel('$X$')
# ax[1,0].set_ylabel('$Y$')
# ax[1,0].legend()
# ax[1,1].plot(t_robot,x_euler_maruyama[:,2]/np.pi, label='Orientation-EM', linewidth=0.5)
# ax[1,1].plot(t_robot,x_linear[:,2]/np.pi, label='Orientation-LIN', linewidth=0.5)
# ax[1,1].plot(t_robot,x_robot_RK[:,2]/np.pi, label='Orientation-RK', linewidth=0.5)
# ax[1,1].set_xlabel('$t$')
# ax[1,1].set_ylabel('$\phi / \pi$')
# ax[1,1].legend()
# 
# 
# =============================================================================

plt.figure()
skip = 30
plt.plot(x_euler_maruyama[:,0],x_euler_maruyama[:,1], '-b', label='Robot-position-EM', linewidth=0.5)
plt.plot(x_linear[:,0],x_linear[:,1], '-r', label='Robot-position-LIN', linewidth=0.5)
plt.plot(x_robot_RK[:,0],x_robot_RK[:,1], '-k', label='Robot-position-RK', linewidth=0.5)
plt.quiver(x_euler_maruyama[::skip,0],x_euler_maruyama[::skip,1],
               np.cos(x_euler_maruyama[::skip,2]),np.sin(x_euler_maruyama[::skip,2]),
               label='Direction-EM', linewidth=0.5, alpha=0.5, color='blue')
plt.quiver(x_linear[::skip,0],x_linear[::skip,1],
               np.cos(x_linear[::skip,2]),np.sin(x_linear[::skip,2]),
               label='Direction-LIN', linewidth=0.5, alpha=0.5, color='red')
plt.quiver(x_robot_RK[::skip,0],x_robot_RK[::skip,1],
               np.cos(x_robot_RK[::skip,2]),np.sin(x_robot_RK[::skip,2]),
               label='Direction-RK', linewidth=0.5, alpha=0.5, color='black')
plt.grid()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.legend()























