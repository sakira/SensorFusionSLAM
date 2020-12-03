#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:16:43 2020
Task 3: Camera calibration
@author: hassans4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

columns = ['true_distance', 
           'height'
          ]

d1 = 1.6 # distance between pinhole and IR detector
d2 = 5 # thickness of the wall
qr_code_length = 11.5 #(cm) # field dimension, 

dataset = 'dataset1'
data_path = '../project_data/' + dataset + '/data/task3/'
data = pd.read_csv(data_path + 'camera_module_calibration_task3.csv', header=None)
data.columns = columns

# save the values
path_to_store = 'results/' + dataset + '/task3/'
filename = path_to_store + 'hieght_vs_distance.pdf'


plt.figure()
plt.plot(1./data.height, data.true_distance + d1 + d2, 'o')
plt.grid()


plt.savefig(filename)
plt.show()

x = 1./data.height
X = x[:, np.newaxis]
n = x.shape[0]
X0 = np.ones((n,1))
X = np.hstack((X, X0))
y = data.true_distance + d1 + d2

m, c = np.linalg.lstsq(X, y, rcond=None)[0]
print('m={}, c={}'.format(m, c))

# focal length
# m = f * qr_code_length

f = m/qr_code_length # focal length in pixel
print('focal length:{}'.format(f))
