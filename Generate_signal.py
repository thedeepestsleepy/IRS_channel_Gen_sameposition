from Generate_signal_func import *
import scipy.io as sio
# Load CSI
# H_G (Num_of_Data,Ni,Nt)
# H_r (Num_of_Data,Num_of_user,Ni) -> single antenna user

hr_train = np.load('./Data/Channel/Channel_Hr.npy')
g_train = np.load('./Data/Channel/Channel_G.npy')
phi_up_init = np.load('./Data/Channel/Phase_uplink_init.npy')
diag_phi_up_init = tf.linalg.diag(phi_up_init)
#shape check (10000, 100, 64) (10000, 4, 100) (10000, 4, 64)
# up-link channel
diag_phi_up_init = tf.cast(diag_phi_up_init,tf.complex64)
hr_train = tf.constant(hr_train,tf.complex64)
g_train = tf.constant(g_train,tf.complex64)
hr_train_real = tf.math.real(hr_train)
hr_train_imag = tf.math.imag(hr_train)
hr_train_AsInput = tf.concat([hr_train_real,hr_train_imag],2)
g_train_real = tf.math.real(g_train)
g_train_imag = tf.math.imag(g_train)
g_train_AsInput = tf.concat([g_train_real,g_train_imag],2)

diag_phi_up_init_real = tf.math.real(diag_phi_up_init)
diag_phi_up_init_imag = tf.math.imag(diag_phi_up_init)

H_train_real = tf.matmul(hr_train_real,diag_phi_up_init_real)
H_train_real = tf.matmul(H_train_real,g_train_real)
H_train_imag = tf.matmul(hr_train_imag,diag_phi_up_init_imag)
H_train_imag = tf.matmul(H_train_imag,g_train_imag)
H_train = tf.complex(H_train_real,H_train_imag)#(10000,4,64)


## combined channel shape is (Num_Data, Num_User, Num_Ant_BS)
H_train_T = tf.transpose(H_train,[0,2,1])#(10000,64,4) up-link


### Generate pilot signals and message signals
counts = np.ones((Num_Data, K*mu))
# Probability of success.
probs = [0.5]
seeds = np.random.randint((1,2))

binomial_samples = tf.random.stateless_binomial(
    shape=[Num_Data,K*mu], seed=seeds, counts=counts, probs=probs)
    #n=1, p=0.5, size=(K * mu, ))
Pilots_matrix_real = Pilots_matrix[:,0:K]
Pilots_matrix_imag = Pilots_matrix[:,K:K*mu]
binomial_samples_real = binomial_samples[:,0:K]
binomial_samples_imag = binomial_samples[:,K:K*mu]


Signal_real = tf.concat([Pilots_matrix_real,binomial_samples_real],1)
Signal_imag = tf.concat([Pilots_matrix_imag,binomial_samples_imag],1)
#Signal [real[pilot,signal],imag[pilot,signal]]
Signal = tf.concat([Signal_real,Signal_imag],1)
code_words = tf.concat([Pilots_matrix,binomial_samples],1)

step = int(Num_Data/100)
signal_y = np.zeros((Num_Data,K*2*2))
for index in range(100):
    print(index)
    signal_y[index*step:(index+1)*step,:]=ofdm_simulate(Signal[index*step:(index+1)*step,:],\
        H_train_T[index*step:(index+1)*step,:,:],step)

signal_y_real = signal_y[:,0:K*mu]

signal_y_imag = signal_y[:,K*mu:K*mu*2]

rec_pilot_real = signal_y_real[:,0:K]

rec_pilot_imag = signal_y_imag[:,0:K]

rec_pilot = tf.concat([rec_pilot_real,rec_pilot_imag],1)

rec_signal_real = signal_y_real[:,K:K*mu]

rec_signal_imag = signal_y_imag[:,K:K*mu]

rec_signal = tf.concat([rec_signal_real,rec_signal_imag],1)
print('end')
print('shape of signal',signal_y.shape)
np.save('./Data/Signal/Received_at_BS.npy',signal_y)
np.save('./Data/Signal/Transmitted_at_User.npy',Signal)
np.save('./Data/Signal/Received_pilot.npy',rec_pilot)
np.save('./Data/Signal/Received_signal.npy',rec_signal)
np.save('./Data/Signal/Transmitted_pilot.npy',Pilots_matrix)
print('pilots shape',Pilots_matrix.shape)
