#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:35:17 2020
Task 4: Determine speed of the robot
@author: hassans4
"""

import pandas as pd

speed_columns = ['distance', 'timelapse']

dataset = 'dataset1'
data_path = '../project_data/' + dataset + '/data/task4/'
task4_data = pd.read_csv(data_path + 'robot_speed_task4.csv', header=None)
task4_data.columns = speed_columns

print(task4_data)
dx = task4_data.distance[1] - task4_data.distance[0]
speed = dx/task4_data.timelapse.mean()
print('speed={} cm/s, dx = {}cm'.format(speed, dx))