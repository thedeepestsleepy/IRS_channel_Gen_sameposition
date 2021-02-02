from __future__ import absolute_import, division, print_function
import tensorflow as tf


from Global_paras import *
print('####START#### TF version :',tf.__version__)

def Modulation_tf(bits):
    b_real = tf.cast(2*bits[:,0:K*mu]-1,tf.float32)
    b_imag = tf.cast(2*bits[:,K*mu:K*mu*2]-1,tf.float32)
    bits_fin = tf.complex(b_real,b_imag)
    return bits_fin
def addCP_tf_2(OFDM_time):
    
    cp = OFDM_time[:,-CP:]               # take the last CP samples ..
    return tf.concat([cp,OFDM_time],1)
def removeCP_tf(signal):
    return signal[:,CP:(CP + K*2)]
def channel_tf_2(signal, channelResponse,bs):
    signal_real = tf.cast(tf.math.real(signal),tf.float32)
    signal_imag = tf.cast(tf.math.imag(signal),tf.float32)
    channelResponse_real = tf.cast(tf.math.real(channelResponse[::-1]),tf.float32)
    channelResponse_imag = tf.cast(tf.math.imag(channelResponse[::-1]),tf.float32)
    
    psnn_tf_conv1d_result_real1 = signalConv1d_func_tf_2_data_gen(signal_real,channelResponse_real,bs)
    psnn_tf_conv1d_result_imag1 = -1*signalConv1d_func_tf_2_data_gen(signal_imag,channelResponse_imag,bs)
    psnn_tf_conv1d_result_real2 = signalConv1d_func_tf_2_data_gen(signal_real,channelResponse_real,bs)
    psnn_tf_conv1d_result_imag2 = signalConv1d_func_tf_2_data_gen(signal_imag,channelResponse_imag,bs)
    psnn_tf_conv1d_result_real = tf.add(psnn_tf_conv1d_result_real1,psnn_tf_conv1d_result_real2)
    psnn_tf_conv1d_result_imag = tf.add(psnn_tf_conv1d_result_imag1,psnn_tf_conv1d_result_imag2)
    psnn_tf_conv1d_result = tf.complex(psnn_tf_conv1d_result_real,psnn_tf_conv1d_result_imag)
    convolved = psnn_tf_conv1d_result
    #print('debug log:convolved iin ch_2',convolved.shape)
    return convolved
def signalConv1d_func_tf_2_data_gen(data,filter,batchS_generate):
    #data : BS,lens     None, 128 -> None,128,1
    # filter: BS,Nt,Nr  None, 4,1-> 4,1,None
    #sess = tf.Session()
    #bs_val = sess.run(batchS)
    # data is 80, filter is 10,3
    #data = tf.expand_dims(data,0)
    data = tf.expand_dims(data,-1)
    filter = tf.transpose(filter,[1,2,0])
    #filter = tf.expand_dims(filter,1)
    #print('\n In signaleConv1d func, data filter shape after expand dims',data.shape,filter.shape)
    
    res = tf.nn.conv1d(data,filter,1,'SAME')
    #print('\n res shape is : \n',res.shape)
    res = tf.cast(res,tf.float32)
    #print('res is :',res)
    #mask = np.zeros([Data_Num,Data_Num,res.shape[1]])
    mask = np.zeros([batchS_generate,batchS_generate,res.shape[1]])
    one = np.ones([1,res.shape[1]])
    for i in range(mask.shape[0]):
        mask[i][i] = one
    mask = np.transpose(mask,(0,2,1))
    mask = tf.convert_to_tensor(mask,tf.float32)
    res_add = res+100
    res_fin = tf.multiply(res_add,mask)
    #print (res_fin.shape,res_fin)
    idx = tf.where(res_fin != 0) 
    output = tf.gather_nd(res_fin, idx)
    output = output - 100
    #print('output tf.gather',output.shape)
    #output = tf.reshape(output,[Data_Num,res.shape[1]])
    output = tf.reshape(output,[batchS_generate,res.shape[1]])
    #print('\n In signaleConv1d func, data filter shape after expand dims',data.shape,filter.shape)
    return output
def ofdm_simulate(codeword, channelResponse,bs):
    # code word shape is [Num_Data, 256]
    codeword_qam = Modulation_tf(codeword)    # shape [Num_data,128] 
    OFDM_time = tf.signal.ifft(codeword_qam) # shape [Num_data,128] 
    OFDM_withCP = addCP_tf_2(OFDM_time) # shape [Num_data,144] 
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel_tf_2(OFDM_TX, channelResponse,bs)
    OFDM_RX_noCP = removeCP_tf(OFDM_RX)
    OFDM_RX_noCP = tf.signal.fft(OFDM_RX_noCP)
    outputs_OFDM_RX_noCP = tf.concat(
        [tf.math.real(OFDM_RX_noCP), tf.math.imag(OFDM_RX_noCP)], 1)

    # # ----- target inputs ---
    # symbol = np.zeros(K, dtype=complex)
    # codeword_qam = Modulation_tf(codeword)
    # #print('codeword_qam',codeword_qam.shape)
    # #symbol[np.arange(K)] = codeword_qam
    # OFDM_data_codeword = codeword_qam
    # #OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    # OFDM_time_codeword = tf.signal.ifft(OFDM_data_codeword)
    # OFDM_withCP_cordword = addCP_tf_2(OFDM_time_codeword)
    # OFDM_RX_codeword = channel_tf(OFDM_withCP_cordword, channelResponse, SNRdb)
    # OFDM_RX_noCP_codeword = removeCP_tf(OFDM_RX_codeword)
    # OFDM_RX_noCP_codeword = DFT_tf(OFDM_RX_noCP_codeword)

    # outputs_OFDM_RX_noCP_codeword = tf.concat(
    #     [tf.math.real(OFDM_RX_noCP_codeword), tf.math.imag(OFDM_RX_noCP_codeword)], 1)
    
    # out_result = tf.concat([outputs_OFDM_RX_noCP,outputs_OFDM_RX_noCP_codeword],1)
    # outputs_signal = tf.cast(out_result, tf.float32)
    return outputs_OFDM_RX_noCP