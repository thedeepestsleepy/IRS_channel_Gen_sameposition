import tensorflow as tf
import numpy as np


phi_up = np.load('./Data/Channel/Phase_uplink_init.npy')
diag_Phi_up = tf.linalg.diag(phi_up)
print(diag_Phi_up.shape)

a= np.array([[1,2,3],[1,2,3]])
b= np.array([[1,2],[1,1],[1,0]])
c =a*b
print(c)