from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
#from Only_paras import *
data_len  = 8
print('##### START #####')
'''
#生成1维复数矩阵
####complex_data
complex_data_real = np.random.binomial(n=1, p=0.5, size=(10, ))
complex_data_imag = np.random.binomial(n=1, p=0.5, size=(10, ))
complex_data_real = complex_data_real.astype(np.float32)
complex_data_imag = complex_data_imag.astype(np.float32)
complex_data = complex_data_real+1j*complex_data_imag
####complex_filter
complex_filter_real = np.random.binomial(n=1, p=0.5, size=(3, ))
complex_filter_imag = np.random.binomial(n=1, p=0.5, size=(3, ))
complex_filter_real = complex_filter_real.astype(np.float32)
complex_filter_imag = complex_filter_imag.astype(np.float32)
complex_filter = complex_filter_real+1j*complex_filter_imag


##print('\n print complex : \n')
##print('\n data: \n',complex_data.shape)
##print('\n filter: \n',complex_filter.shape)

#convolution with np.convolve

np_convolve_result = np.convolve(complex_data,complex_filter[::-1],'VALID')

print('\n np.convolve result shape is : \n',np_convolve_result.shape,'\n np.convolve result is : \n')
print(np_convolve_result)
#维度转换 用于tf.nn.conv1d
#data_real 转换为维度[number,width,high]
complex_data_real = tf.expand_dims(complex_data_real,0)
complex_data_real = tf.expand_dims(complex_data_real,-1)
complex_data_imag = tf.expand_dims(complex_data_imag,0)
complex_data_imag = tf.expand_dims(complex_data_imag,-1)
#filter_imag
complex_filter_real = tf.expand_dims(complex_filter_real,-1)
complex_filter_real = tf.expand_dims(complex_filter_real,-1)
complex_filter_imag = tf.expand_dims(complex_filter_imag,-1)
complex_filter_imag = tf.expand_dims(complex_filter_imag,-1)
#print('\n data reshape result: \n', complex_data_real.shape,complex_filter_real.shape)
tf_conv1d_result_real_1 = tf.nn.conv1d(complex_data_real,complex_filter_real,1,'SAME')
tf_conv1d_result_real_2 = -1*tf.nn.conv1d(complex_data_imag,complex_filter_imag,1,'SAME')
tf_conv1d_result_imag_1 = tf.nn.conv1d(complex_data_real,complex_filter_imag,1,'SAME')
tf_conv1d_result_imag_2 = tf.nn.conv1d(complex_data_imag,complex_filter_real,1,'SAME')
tf_conv1d_result_real = tf_conv1d_result_real_1 + tf_conv1d_result_real_2
tf_conv1d_result_imag = tf_conv1d_result_imag_1 + tf_conv1d_result_imag_2
tf_conv1d_result = tf.complex(tf_conv1d_result_real,tf_conv1d_result_imag)
tf_conv1d_result = tf.squeeze(tf_conv1d_result)

print('\n tf.nn.conv1d result shape is : \n',tf_conv1d_result.shape,'\n tf.nn.conv1d result is : \n',tf_conv1d_result)

'''
#####

print('##### 10组数据测试 #####')

#生成1维复数矩阵

def signalConv1d_func(data,filter):
    data = tf.expand_dims(data,-1)
    filter = tf.transpose(filter,[1,0])
    filter = tf.expand_dims(filter,1)
    #print('\n In signaleConv1d func, data filter shape after expand dims',data.shape,filter.shape)
    
    res = tf.nn.conv1d(data,filter,1,'SAME')
    print('\n res shape is : \n',res.shape)
    res = tf.cast(res,tf.float32)
    print('res is :',res)
    mask = np.zeros([10,10,10])
    one = np.ones([1,res.shape[1]])
    for i in range(mask.shape[0]):
        mask[i][i] = one
    mask = np.transpose(mask,(0,2,1))
    mask = tf.convert_to_tensor(mask,tf.float32)
    res_add = res+mask
    res_fin = tf.multiply(res_add,mask)
    #print (res_fin.shape,res_fin)
    idx = tf.where(res_fin != 0)
    output = tf.gather_nd(res_fin, idx)
    output = output - 1
    #print('output tf.gather',output.shape)
    output = tf.reshape(output,[10,res.shape[1]])
    #print('\n In signaleConv1d func, data filter shape after expand dims',data.shape,filter.shape)
    return output

def signalConv1d_func_2(data,filter):
    # data is 80, filter is 10,3
    data = tf.expand_dims(data,0)
    data = tf.expand_dims(data,-1)
    filter = tf.transpose(filter,[1,0])
    filter = tf.expand_dims(filter,1)
    #print('\n In signaleConv1d func, data filter shape after expand dims',data.shape,filter.shape)
    
    res = tf.nn.conv1d(data,filter,1,'SAME')
    #print('\n res shape is : \n',res.shape)
    res = tf.cast(res,tf.float32)
    #print('res is :',res)
    mask = np.zeros([Data_Num,Data_Num,data_len])
    one = np.ones([1,data_len])
    for i in range(mask.shape[0]):
        mask[i][i] = one
    mask = np.transpose(mask,(0,2,1))
    mask = tf.convert_to_tensor(mask,tf.float32)
    res_add = res+1
    res_fin = tf.multiply(res_add,mask)
    #print (res_fin.shape,res_fin)
    idx = tf.where(res_fin != 0) 
    output = tf.gather_nd(res_fin, idx)
    output = output - 1
    #print('output tf.gather',output.shape)
    output = tf.reshape(output,[10,res.shape[1]])
    #print('\n In signaleConv1d func, data filter shape after expand dims',data.shape,filter.shape)
    return output
Data_Num = 20

####complex_data
complex_data_real = np.random.binomial(n=1, p=0.5, size=(data_len,))
complex_data_imag = np.random.binomial(n=1, p=0.5, size=(data_len,))
complex_data_real = complex_data_real.astype(np.float32)
complex_data_imag = complex_data_imag.astype(np.float32)
complex_data = complex_data_real+1j*complex_data_imag
####complex_filter
complex_filter_real = np.random.binomial(n=1, p=0.5, size=(Data_Num,3))
complex_filter_imag = np.random.binomial(n=1, p=0.5, size=(Data_Num,3))
complex_filter_real = complex_filter_real.astype(np.float32)
complex_filter_imag = complex_filter_imag.astype(np.float32)
complex_filter = complex_filter_real+1j*complex_filter_imag

#print('\n print complex : \n')
#print('\n data: \n',complex_data.shape)
#print('\n filter: \n',complex_filter.shape)

#convolution with np.convolve
conv_10_test_result = []
for index in range(10):
    conv_10_test_result.append(np.convolve(complex_data,complex_filter[index,::-1],'SAME'))
#print('np.concolve result for 10data is \n',np.array(conv_10_test_result).shape)
print('np result 1 ',np.array(conv_10_test_result))
 
#### 

print('##### START test tf.conv1d for 10 datasets')
print('#####')


conv_10_test_result_1 = signalConv1d_func_2(complex_data_real,complex_filter_real)
conv_10_test_result_2 = signalConv1d_func_2(complex_data_real,complex_filter_imag)
conv_10_test_result_3 = signalConv1d_func_2(complex_data_imag,complex_filter_real)
conv_10_test_result_4 = -1*signalConv1d_func_2(complex_data_imag,complex_filter_imag)
tf_conv1d_result_real = conv_10_test_result_1 + conv_10_test_result_4
tf_conv1d_result_imag = conv_10_test_result_2 + conv_10_test_result_3
tf_conv1d_result = tf.complex(tf_conv1d_result_real,tf_conv1d_result_imag)
tf_conv1d_result = tf.squeeze(tf_conv1d_result)
print('tf.concolve result for 10data is \n',np.array(tf_conv1d_result).shape)
print('tf conv result 1 ',np.array(tf_conv1d_result))

####
'''
bits_ifft_test = np.random.binomial(n=1, p=0.5, size=(10, ))
ifft_test_result = np.fft.ifft(bits_ifft_test)
fft_test_result = np.fft.fft(bits_ifft_test)
print('test ifft',ifft_test_result,fft_test_result)
ifft_test_tf_result = tf.signal.ifft(bits_ifft_test)
fft_test_tf_result = tf.signal.fft(bits_ifft_test)
print('test ifft',ifft_test_tf_result,fft_test_tf_result)

###
print('test multy prod single')
channel_train = np.load('channel_train.npy')
train_size = channel_train.shape[0]
channel_test = np.load('channel_test.npy')
test_size = channel_test.shape[0]
index = np.random.choice(np.arange(train_size), size=10)
H_total = channel_train[index]

print('input data shape:',channel_train.shape)
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
#print('train data input', bits.shape,H.shape)
signal_output, para = ofdm_simulate(bits, H_total[0], SNRdb)
print('debug log: result',signal_output.shape)
'''